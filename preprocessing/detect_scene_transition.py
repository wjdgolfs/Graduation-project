from pathlib import Path
import csv

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


def calculate_frame_difference(previous_frame, current_frame):
    previous_gray = cv2.cvtColor(
        previous_frame,
        cv2.COLOR_BGR2GRAY
    )

    current_gray = cv2.cvtColor(
        current_frame,
        cv2.COLOR_BGR2GRAY
    )

    difference = cv2.absdiff(
        previous_gray,
        current_gray
    )

    score = np.mean(difference) / 255.0

    return float(score)


def group_transition_frames(
    frame_scores,
    gradual_threshold
):
    groups = []
    current_group = []

    for frame_number, score in frame_scores:
        if score >= gradual_threshold:
            current_group.append(
                (frame_number, score)
            )
        else:
            if current_group:
                groups.append(current_group)
                current_group = []

    if current_group:
        groups.append(current_group)

    return groups


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

        if (
            length <= 2
            and max_score >= hard_cut_threshold
        ):
            transition_type = "HARD_CUT"

        elif length >= min_gradual_frames:
            transition_type = "GRADUAL"

        else:
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

    success, previous_frame = capture.read()

    if not success:
        capture.release()

        raise RuntimeError(
            "Failed to read first frame."
        )

    frame_number = 1
    frame_scores = []

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


def main():
    config = load_config()

    video_path = (
        Path(config["paths"]["temp"])
        / "scene_transition_test.mp4"
    )

    detect_scene_transitions(
        video_path
    )


if __name__ == "__main__":
    main()