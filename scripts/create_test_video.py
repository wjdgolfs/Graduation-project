# 장면이 바뀌는 시점을 미리 아는 테스트 영상을 직접 만듭니다.
# 실제 영상보다 결과를 비교하기 쉬워 장면 전환 탐지 방식을 점검하는 데 사용합니다.
# 입력: config.yaml의 temp 경로 / 출력: 640×360, 30 FPS, 9초짜리 MP4.
# A→B는 즉시 전환, B→검정→C는 페이드, C→D는 두 화면을 섞는 디졸브입니다.

from pathlib import Path

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


# 어두운 배경에 밝은 사각형을 그린 A 장면 이미지를 반환합니다.
def create_scene_a(width, height):
    # 이미지는 (높이, 너비, 색상 3채널) 배열입니다. uint8은 픽셀 값을 0~255 정수로 저장합니다.
    frame = np.full(
        (height, width, 3),
        40,
        dtype=np.uint8
    )

    # 좌표는 (가로 x, 세로 y), 색은 OpenCV의 BGR 순서입니다. 두께 -1은 내부 채우기입니다.
    cv2.rectangle(
        frame,
        (80, 100),
        (280, 260),
        (200, 200, 200),
        -1
    )

    # 화면에 장면 이름을 적어 영상을 직접 볼 때 전환 전후를 쉽게 구분합니다.
    cv2.putText(
        frame,
        "SCENE A",
        (220, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    return frame


# 밝은 배경에 어두운 원을 그려 A와 밝기가 크게 다른 B 장면을 만듭니다.
def create_scene_b(width, height):
    frame = np.full(
        (height, width, 3),
        180,
        dtype=np.uint8
    )

    cv2.circle(
        frame,
        (450, 180),
        80,
        (40, 40, 40),
        -1
    )

    cv2.putText(
        frame,
        "SCENE B",
        (220, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (20, 20, 20),
        2
    )

    return frame


# 중간 밝기 배경에 사각 테두리를 그려 페이드 후 나타날 C 장면을 만듭니다.
def create_scene_c(width, height):
    frame = np.full(
        (height, width, 3),
        90,
        dtype=np.uint8
    )

    cv2.rectangle(
        frame,
        (180, 80),
        (460, 280),
        (220, 220, 220),
        5
    )

    cv2.putText(
        frame,
        "SCENE C",
        (220, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    return frame


# 밝은 배경과 대각선으로 C와 구별되는 D 장면을 만듭니다.
def create_scene_d(width, height):
    frame = np.full(
        (height, width, 3),
        210,
        dtype=np.uint8
    )

    cv2.line(
        frame,
        (100, 280),
        (540, 80),
        (30, 30, 30),
        12
    )

    cv2.putText(
        frame,
        "SCENE D",
        (220, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (20, 20, 20),
        2
    )

    return frame


# 같은 이미지를 frame_count번 기록해 정지 장면을 일정 시간 유지합니다.
def write_static_scene(writer, frame, frame_count):
    # _는 반복 횟수만 필요하고 반복 번호는 쓰지 않는다는 뜻입니다.
    for _ in range(frame_count):
        writer.write(frame)


# 시작 이미지의 비중을 줄이고 끝 이미지의 비중을 늘려 부드러운 전환을 기록합니다.
def write_fade(writer, start_frame, end_frame, frame_count):
    for index in range(frame_count):
        # 혼합 비율 alpha를 0에서 1까지 늘립니다. max(..., 1)은 프레임 수가 1일 때 0으로 나누지 않게 합니다.
        alpha = index / max(
            frame_count - 1,
            1
        )

        # 결과 = 시작 × (1-alpha) + 끝 × alpha입니다. 검정과 섞으면 페이드가 됩니다.
        frame = cv2.addWeighted(
            start_frame,
            1.0 - alpha,
            end_frame,
            alpha,
            0
        )

        writer.write(frame)


# 설정 경로에 장면 A~D를 조합한 테스트 영상을 저장하고 구성 시간을 출력합니다.
def create_test_video():
    config = load_config()

    output_dir = Path(config["paths"]["temp"])

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = output_dir / "scene_transition_test.mp4"

    width = 640
    height = 360
    fps = 30

    scene_a = create_scene_a(width, height)
    scene_b = create_scene_b(width, height)
    scene_c = create_scene_c(width, height)
    scene_d = create_scene_d(width, height)

    # 모든 채널이 0인 검정 이미지를 만들어 페이드 아웃/인의 중간 화면으로 씁니다.
    black_frame = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    # mp4v는 영상을 압축해 저장할 코덱 식별자입니다.
    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    # 프레임 배열은 (높이, 너비, 채널)이지만 영상 크기 인자는 (너비, 높이) 순서입니다.
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Failed to create video file."
        )

    # 0~2초: A를 60장 씁니다. 이어 B를 쓰면 중간 혼합 없이 하드 컷이 생깁니다.
    write_static_scene(
        writer,
        scene_a,
        fps * 2
    )

    # 2~4초: B를 유지합니다.
    write_static_scene(
        writer,
        scene_b,
        fps * 2
    )

    # 4~5초: B에서 검정으로 어두워지는 페이드 아웃입니다.
    write_fade(
        writer,
        scene_b,
        black_frame,
        fps
    )

    # 5~6초: 검정에서 C가 나타나는 페이드 인입니다.
    write_fade(
        writer,
        black_frame,
        scene_c,
        fps
    )

    # 6~7초: C를 유지합니다.
    write_static_scene(
        writer,
        scene_c,
        fps
    )

    # 7~8초: 검정을 거치지 않고 C와 D를 섞는 디졸브입니다.
    write_fade(
        writer,
        scene_c,
        scene_d,
        fps
    )

    # 8~9초: D를 유지합니다.
    write_static_scene(
        writer,
        scene_d,
        fps
    )

    # 파일 기록을 마무리하고 영상 저장 자원을 반환합니다.
    writer.release()

    total_frames = fps * 9

    print("-" * 50)
    print("Test video created")
    print("Output:", output_path)
    print("Resolution:", f"{width}x{height}")
    print("FPS:", fps)
    print("Frames:", total_frames)
    print("Duration:", total_frames / fps, "sec")
    print("-" * 50)
    print("0 - 2 sec : Scene A")
    print("2 sec     : Hard Cut")
    print("2 - 4 sec : Scene B")
    print("4 - 6 sec : Fade")
    print("6 - 7 sec : Scene C")
    print("7 - 8 sec : Dissolve")
    print("8 - 9 sec : Scene D")
    print("-" * 50)


# 다른 파일에서 import할 때는 실행하지 않고, 이 파일을 직접 실행했을 때만 아래 작업을 시작합니다.
if __name__ == "__main__":
    create_test_video()
