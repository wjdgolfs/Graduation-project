# 연속한 두 프레임의 밝기 차이로 장면이 바뀌는 후보 구간을 찾습니다.
# 갑작스러운 전환(HARD_CUT)과 점진적 전환(GRADUAL)을 간단한 규칙으로 구분합니다.
# 입력: 테스트 영상 / 출력: 프레임별 차이 CSV와 콘솔의 후보 구간 목록.
# 학습된 AI 모델은 사용하지 않습니다. 물체 이동이나 조명 변화도 후보로 잡힐 수 있습니다.

from pathlib import Path
import csv

import cv2
import numpy as np
import yaml


# __file__는 현재 파일 경로입니다. 상위 두 단계인 프로젝트 폴더를 기준으로 설정 파일을 찾습니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


# 여러 스크립트가 같은 경로와 기준값을 쓰도록 YAML 설정을 딕셔너리로 읽습니다.
def load_config():
    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return yaml.safe_load(file)


# 이전/현재 BGR 이미지의 평균 밝기 차이를 0~1 범위의 실수로 반환합니다.
def calculate_frame_difference(previous_frame, current_frame):
    # 색상 세 채널을 밝기 한 채널로 바꿔 계산을 단순화합니다. 색만 달라지는 전환은 놓칠 수 있습니다.
    previous_gray = cv2.cvtColor(
        previous_frame,
        cv2.COLOR_BGR2GRAY
    )

    current_gray = cv2.cvtColor(
        current_frame,
        cv2.COLOR_BGR2GRAY
    )

    # 같은 위치 픽셀끼리 밝기를 빼고 절댓값을 취하므로 밝아짐과 어두워짐 모두 변화로 셉니다.
    difference = cv2.absdiff(
        previous_gray,
        current_gray
    )

    # 픽셀 차이의 평균을 255로 나눕니다. 0은 밝기 변화 없음, 1은 가능한 최대 평균 차이입니다.
    score = np.mean(difference) / 255.0

    return float(score)


# 시간순 (프레임 번호, 점수) 목록에서 기준 이상인 연속 항목을 묶어 반환합니다.
def group_transition_frames(
    frame_scores,
    gradual_threshold
):
    groups = []
    current_group = []

    # 기준 이상이 이어지는 동안 같은 후보 구간으로 모읍니다.
    for frame_number, score in frame_scores:
        if score >= gradual_threshold:
            current_group.append(
                (frame_number, score)
            )
        else:
            if current_group:
                groups.append(current_group)
                current_group = []

    # 영상 끝까지 변화가 이어진 마지막 구간도 빠뜨리지 않고 추가합니다.
    if current_group:
        groups.append(current_group)

    return groups


# 후보 구간마다 길이와 최대 점수를 보고 종류·시작·끝 등을 담은 딕셔너리를 반환합니다.
def classify_transitions(
    groups,
    hard_cut_threshold,
    min_gradual_frames
):
    transitions = []

    for group in groups:
        frame_numbers = [
            item[0]
            for item in group
        ]

        scores = [
            item[1]
            for item in group
        ]

        start_frame = frame_numbers[0]
        end_frame = frame_numbers[-1]

        max_score = max(scores)
        length = len(group)

        # 1~2프레임의 짧은 구간이며 최대 점수가 높으면 급격한 전환으로 분류합니다.
        if (
            length <= 2
            and max_score >= hard_cut_threshold
        ):
            transition_type = "HARD_CUT"

        # 점수가 낮은 기준 이상으로 충분히 오래 이어지면 점진적 전환으로 분류합니다.
        elif length >= min_gradual_frames:
            transition_type = "GRADUAL"

        else:
            # 두 조건을 충족하지 못한 후보도 버리지 않고 작은 변화로 남깁니다.
            transition_type = "SMALL_CHANGE"

        transitions.append(
            {
                "type": transition_type,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "length": length,
                "max_score": max_score
            }
        )

    return transitions


# 프레임 번호와 변화 점수를 CSV로 저장해 시각화 및 장면 번호 생성에서 재사용합니다.
def save_scores(
    frame_scores,
    output_path
):
    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "frame_number",
                "difference_score"
            ]
        )

        for frame_number, score in frame_scores:
            writer.writerow(
                [
                    frame_number,
                    score
                ]
            )


# 영상을 순서대로 읽고 점수 계산 → 후보 묶기 → 분류 → CSV 저장 및 요약 출력을 수행합니다.
def detect_scene_transitions(video_path):
    config = load_config()

    video_path = Path(video_path)

    hard_cut_threshold = (
        config["scene_detection"]["hard_cut_threshold"]
    )

    gradual_threshold = (
        config["scene_detection"]["gradual_threshold"]
    )

    min_gradual_frames = (
        config["scene_detection"]["min_gradual_frames"]
    )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Failed to open video: {video_path}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    # 첫 프레임은 비교 기준입니다. 이전 이미지가 없으므로 자체 점수는 기록하지 않습니다.
    success, previous_frame = capture.read()

    if not success:
        capture.release()

        raise RuntimeError(
            "Failed to read first frame."
        )

    # 번호는 0부터 시작합니다. 첫 점수의 번호 1은 0번과 1번 프레임의 차이를 뜻합니다.
    frame_number = 1
    frame_scores = []

    # 영상 끝 또는 읽기 실패까지 순차 처리합니다. JPG 추출 개수 설정은 여기서 사용하지 않습니다.
    while True:
        success, current_frame = capture.read()

        if not success:
            break

        score = calculate_frame_difference(
            previous_frame,
            current_frame
        )

        frame_scores.append(
            (
                frame_number,
                score
            )
        )

        # 현재 이미지를 다음 반복의 이전 이미지로 넘겨 이웃한 두 프레임만 비교합니다.
        previous_frame = current_frame
        frame_number += 1

    capture.release()

    groups = group_transition_frames(
        frame_scores,
        gradual_threshold
    )

    transitions = classify_transitions(
        groups,
        hard_cut_threshold,
        min_gradual_frames
    )

    results_dir = Path(
        config["paths"]["results"]
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    score_output_path = (
        results_dir
        / f"{video_path.stem}_scene_scores.csv"
    )

    save_scores(
        frame_scores,
        score_output_path
    )

    print("-" * 60)
    print("Scene transition detection")
    print("Video:", video_path)
    print("FPS:", round(fps, 2))
    print("-" * 60)

    for index, transition in enumerate(
        transitions,
        start=1
    ):
        # 프레임 번호 ÷ FPS로 초 단위 위치를 계산합니다. 정상적인 양수 FPS를 전제로 합니다.
        start_time = (
            transition["start_frame"]
            / fps
        )

        end_time = (
            transition["end_frame"]
            / fps
        )

        print(
            f"[{index}] "
            f"{transition['type']} | "
            f"frame "
            f"{transition['start_frame']} "
            f"- "
            f"{transition['end_frame']} | "
            f"time "
            f"{start_time:.2f}s "
            f"- "
            f"{end_time:.2f}s | "
            f"max score: "
            f"{transition['max_score']:.4f}"
        )

    print("-" * 60)
    print(
        "Detected transitions:",
        len(transitions)
    )
    print(
        "Score CSV:",
        score_output_path
    )
    print("-" * 60)


# 이 파일을 직접 실행할 때 사용할 입력 경로와 설정을 준비하고 작업을 시작합니다.
def main():
    config = load_config()

    video_path = (
        Path(config["paths"]["temp"])
        / "scene_transition_test.mp4"
    )

    detect_scene_transitions(
        video_path
    )


# 다른 파일에서 import할 때는 실행하지 않고, 이 파일을 직접 실행했을 때만 아래 작업을 시작합니다.
if __name__ == "__main__":
    main()
