from pathlib import Path

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config():
    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return yaml.safe_load(file)


def extract_frames(video_path, output_root, num_frames):
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Failed to open video: {video_path}"
        )

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

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        num_frames,
        dtype=int
    )

    output_dir = (
        Path(output_root)
        / video_path.stem
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    saved_count = 0

    for order, frame_index in enumerate(frame_indices):
        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index)
        )

        success, frame = capture.read()

        if not success:
            continue

        output_path = (
            output_dir
            / f"frame_{order:04d}.jpg"
        )

        cv2.imwrite(
            str(output_path),
            frame
        )

        saved_count += 1

        print(
            f"[{saved_count}/{num_frames}] "
            f"frame {frame_index} "
            f"-> {output_path.name}"
        )

    capture.release()

    print("-" * 50)
    print("Saved frames:", saved_count)
    print("Output:", output_dir)
    print("-" * 50)


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


if __name__ == "__main__":
    main()