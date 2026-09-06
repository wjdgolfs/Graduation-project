# 프레임별 차이 점수를 후속 처리에서 쓰기 쉬운 장면 전환 번호로 바꿉니다.
# 입력: detect_scene_transition.py의 점수 CSV / 출력: 장면 번호와 종류를 추가한 CSV.
# 번호의 의미: 0 = 일반, 1 = 급격한 전환, 2 = 점진적인 전환입니다.
# 여기서는 큰 점수를 프레임별로 즉시 판정하므로, 구간 길이도 보는 탐지 코드와 결과가 다를 수 있습니다.

from pathlib import Path
import csv

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


# 점수 CSV를 읽어 프레임 번호(int)와 점수(float)를 가진 딕셔너리 목록으로 반환합니다.
def load_scene_scores(csv_path):
    frame_scores = []

    with open(
        csv_path,
        "r",
        encoding="utf-8"
    ) as file:
        # CSV 첫 줄을 열 이름으로 사용합니다. 읽은 값은 문자열이므로 아래에서 숫자로 바꿉니다.
        reader = csv.DictReader(file)

        for row in reader:
            frame_scores.append(
                {
                    "frame_number": int(row["frame_number"]),
                    "difference_score": float(
                        row["difference_score"]
                    )
                }
            )

    return frame_scores


# 시간순 점수 목록과 세 기준값을 받아, 각 항목에 전환 번호와 이름을 붙인 목록을 반환합니다.
def create_scene_signal(
    frame_scores,
    hard_cut_threshold,
    gradual_threshold,
    min_gradual_frames
):
    scene_signals = []

    # 점진적 전환 후보를 임시 보관합니다. 충분히 이어지는지 알아야 최종 판정할 수 있습니다.
    gradual_buffer = []

    for item in frame_scores:
        frame_number = item["frame_number"]
        difference_score = item["difference_score"]

        # 큰 점수는 구간 길이와 관계없이 즉시 급격한 전환(1)으로 기록합니다.
        if difference_score >= hard_cut_threshold:
            scene_signals.append(
                {
                    "frame_number": frame_number,
                    "difference_score": difference_score,
                    "scene_signal": 1,
                    "scene_type": "HARD_CUT"
                }
            )

            # 현재 구현은 큰 변화 직전 후보를 확정하지 않고 비웁니다. 해당 후보들은 NORMAL로 남습니다.
            gradual_buffer = []

        # 낮은 기준 이상이면서 높은 기준 미만인 점수는 점진적 전환 후보로 모읍니다.
        elif difference_score >= gradual_threshold:
            gradual_buffer.append(
                {
                    "frame_number": frame_number,
                    "difference_score": difference_score
                }
            )

            # 후보 길이가 아직 확정되지 않았으므로 일단 일반(0)으로 기록하고 나중에 수정합니다.
            scene_signals.append(
                {
                    "frame_number": frame_number,
                    "difference_score": difference_score,
                    "scene_signal": 0,
                    "scene_type": "NORMAL"
                }
            )

        else:
            # 작은 점수가 나오면 후보 구간이 끝납니다. 최소 길이를 채웠을 때만 점진적 전환으로 확정합니다.
            if len(gradual_buffer) >= min_gradual_frames:
                # 집합(set)에 후보 번호를 모아 어떤 기존 결과를 수정해야 하는지 확인합니다.
                gradual_frame_numbers = {
                    item["frame_number"]
                    for item in gradual_buffer
                }

                for signal in scene_signals:
                    if (
                        signal["frame_number"]
                        in gradual_frame_numbers
                    ):
                        signal["scene_signal"] = 2
                        signal["scene_type"] = "GRADUAL"

            gradual_buffer = []

            scene_signals.append(
                {
                    "frame_number": frame_number,
                    "difference_score": difference_score,
                    "scene_signal": 0,
                    "scene_type": "NORMAL"
                }
            )

    # CSV 끝에서 종료된 후보도 같은 규칙으로 확정합니다. 입력은 연속된 프레임 순서를 전제로 합니다.
    if len(gradual_buffer) >= min_gradual_frames:
        gradual_frame_numbers = {
            item["frame_number"]
            for item in gradual_buffer
        }

        for signal in scene_signals:
            if (
                signal["frame_number"]
                in gradual_frame_numbers
            ):
                signal["scene_signal"] = 2
                signal["scene_type"] = "GRADUAL"

    return scene_signals


# 점수와 전환 번호/이름을 CSV로 저장합니다. 같은 경로의 기존 파일은 덮어씁니다.
def save_scene_signals(
    scene_signals,
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
                "difference_score",
                "scene_signal",
                "scene_type"
            ]
        )

        for item in scene_signals:
            writer.writerow(
                [
                    item["frame_number"],
                    item["difference_score"],
                    item["scene_signal"],
                    item["scene_type"]
                ]
            )


# 장면 전환의 개수가 아니라 각 종류로 표시된 프레임 수를 세어 출력합니다.
def print_summary(scene_signals):
    normal_count = 0
    hard_cut_count = 0
    gradual_count = 0

    for item in scene_signals:
        scene_type = item["scene_type"]

        if scene_type == "NORMAL":
            normal_count += 1

        elif scene_type == "HARD_CUT":
            hard_cut_count += 1

        elif scene_type == "GRADUAL":
            gradual_count += 1

    print("-" * 60)
    print("Scene Signal Summary")
    print("-" * 60)
    print("Normal frames:", normal_count)
    print("Hard Cut frames:", hard_cut_count)
    print("Gradual frames:", gradual_count)
    print("-" * 60)


# 이 파일을 직접 실행할 때 사용할 입력 경로와 설정을 준비하고 작업을 시작합니다.
def main():
    config = load_config()

    results_dir = Path(
        config["paths"]["results"]
    )

    # 탐지 스크립트가 만든 CSV가 먼저 있어야 합니다. 이 작업은 영상 자체를 다시 읽지 않습니다.
    input_path = (
        results_dir
        / "scene_transition_test_scene_scores.csv"
    )

    output_path = (
        results_dir
        / "scene_transition_test_scene_signals.csv"
    )

    hard_cut_threshold = (
        config["scene_detection"]["hard_cut_threshold"]
    )

    gradual_threshold = (
        config["scene_detection"]["gradual_threshold"]
    )

    min_gradual_frames = (
        config["scene_detection"]["min_gradual_frames"]
    )

    frame_scores = load_scene_scores(
        input_path
    )

    scene_signals = create_scene_signal(
        frame_scores,
        hard_cut_threshold,
        gradual_threshold,
        min_gradual_frames
    )

    save_scene_signals(
        scene_signals,
        output_path
    )

    print_summary(
        scene_signals
    )

    print(
        "Output:",
        output_path
    )


# 다른 파일에서 import할 때는 실행하지 않고, 이 파일을 직접 실행했을 때만 아래 작업을 시작합니다.
if __name__ == "__main__":
    main()
