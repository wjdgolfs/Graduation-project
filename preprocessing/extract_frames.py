# 영상 전체에서 일정한 간격으로 이미지를 골라 JPG로 저장하는 코드입니다.
# 모든 프레임을 저장하지 않고 적은 수의 이미지로 영상 내용을 살펴보기 위해 사용합니다.
# 입력: temp 폴더의 테스트 영상 / 출력: frames/영상이름/frame_0000.jpg 등.
# 장면 전환 탐지와는 별도 작업이며, 탐지 코드는 추출한 JPG가 아니라 원본 영상을 읽습니다.

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


# video_path에서 num_frames개 위치를 골라 output_root 아래에 저장합니다. 반환값은 없습니다.
def extract_frames(video_path, output_root, num_frames):
    # 문자열 경로를 Path 객체로 바꾸면 파일 존재 확인과 경로 결합이 편리합니다.
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    # OpenCV로 영상을 엽니다. 경로가 있어도 코덱 등의 문제로 열지 못할 수 있습니다.
    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Failed to open video: {video_path}"
        )

    # 프레임은 영상의 한 장면 이미지, FPS는 1초에 표시하는 프레임 수입니다.
    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    print("-" * 50)
    print("Video:", video_path)
    print("Total frames:", total_frames)
    print("FPS:", round(fps, 2))
    print("-" * 50)

    # 첫 프레임(0)부터 마지막 프레임까지 균등하게 위치를 고릅니다.
    # 요청 수가 전체 프레임 수보다 크면 정수 변환 때문에 같은 위치가 중복될 수 있습니다.
    frame_indices = np.linspace(
        0,
        total_frames - 1,
        num_frames,
        dtype=int
    )

    # stem은 확장자를 뺀 파일 이름입니다. 영상마다 별도 폴더에 결과를 모읍니다.
    output_dir = (
        Path(output_root)
        / video_path.stem
    )

    # 중간 폴더까지 만들며, 이미 있는 폴더는 그대로 사용합니다.
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    saved_count = 0

    # order는 저장 순서(0부터), frame_index는 원본 영상의 프레임 위치입니다.
    for order, frame_index in enumerate(frame_indices):
        # 해당 프레임 위치로 이동한 다음 한 장을 읽습니다.
        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index)
        )

        success, frame = capture.read()

        # 읽기 실패한 프레임은 건너뛰므로 실제 저장 수가 요청 수보다 작을 수 있습니다.
        if not success:
            continue

        # 04d는 번호를 0000처럼 네 자리로 맞춥니다. 같은 이름의 파일은 덮어씁니다.
        output_path = (
            output_dir
            / f"frame_{order:04d}.jpg"
        )

        cv2.imwrite(
            str(output_path),
            frame
        )

        # 현재 코드는 읽기에 성공한 횟수를 셉니다. imwrite의 저장 성공 여부는 별도로 확인하지 않습니다.
        saved_count += 1

        print(
            f"[{saved_count}/{num_frames}] "
            f"frame {frame_index} "
            f"-> {output_path.name}"
        )

    # 영상 파일을 읽는 데 사용한 자원을 반환합니다.
    capture.release()

    print("-" * 50)
    print("Saved frames:", saved_count)
    print("Output:", output_dir)
    print("-" * 50)


# 이 파일을 직접 실행할 때 사용할 입력 경로와 설정을 준비하고 작업을 시작합니다.
def main():
    config = load_config()

    video_path = (
        Path(config["paths"]["temp"])
        / "scene_transition_test.mp4"
    )

    output_root = config["paths"]["frames"]
    num_frames = config["preprocessing"]["num_frames"]

    extract_frames(
        video_path,
        output_root,
        num_frames
    )


# 다른 파일에서 import할 때는 실행하지 않고, 이 파일을 직접 실행했을 때만 아래 작업을 시작합니다.
if __name__ == "__main__":
    main()
