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


def create_scene_a(width, height):
    frame = np.full(
        (height, width, 3),
        40,
        dtype=np.uint8
    )

    cv2.rectangle(
        frame,
        (80, 100),
        (280, 260),
        (200, 200, 200),
        -1
    )

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


def write_static_scene(writer, frame, frame_count):
    for _ in range(frame_count):
        writer.write(frame)


def write_fade(writer, start_frame, end_frame, frame_count):
    for index in range(frame_count):
        alpha = index / max(
            frame_count - 1,
            1
        )

        frame = cv2.addWeighted(
            start_frame,
            1.0 - alpha,
            end_frame,
            alpha,
            0
        )

        writer.write(frame)


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

    black_frame = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

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

    write_static_scene(
        writer,
        scene_a,
        fps * 2
    )

    write_static_scene(
        writer,
        scene_b,
        fps * 2
    )

    write_fade(
        writer,
        scene_b,
        black_frame,
        fps
    )

    write_fade(
        writer,
        black_frame,
        scene_c,
        fps
    )

    write_static_scene(
        writer,
        scene_c,
        fps
    )

    write_fade(
        writer,
        scene_c,
        scene_d,
        fps
    )

    write_static_scene(
        writer,
        scene_d,
        fps
    )

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


if __name__ == "__main__":
    create_test_video()