from pathlib import Path
import csv

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


def load_scene_scores(csv_path):
    frame_scores = []

    with open(
        csv_path,
        "r",
        encoding="utf-8"
    ) as file:
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


def create_scene_signal(
    frame_scores,
    hard_cut_threshold,
    gradual_threshold,
    min_gradual_frames
):
    scene_signals = []

    gradual_buffer = []

    for item in frame_scores:
        frame_number = item["frame_number"]
        difference_score = item["difference_score"]

        if difference_score >= hard_cut_threshold:
            scene_signals.append(
                {
                    "frame_number": frame_number,
                    "difference_score": difference_score,
                    "scene_signal": 1,
                    "scene_type": "HARD_CUT"
                }
            )

            gradual_buffer = []

        elif difference_score >= gradual_threshold:
            gradual_buffer.append(
                {
                    "frame_number": frame_number,
                    "difference_score": difference_score
                }
            )

            scene_signals.append(
                {
                    "frame_number": frame_number,
                    "difference_score": difference_score,
                    "scene_signal": 0,
                    "scene_type": "NORMAL"
                }
            )

        else:
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

            gradual_buffer = []

            scene_signals.append(
                {
                    "frame_number": frame_number,
                    "difference_score": difference_score,
                    "scene_signal": 0,
                    "scene_type": "NORMAL"
                }
            )

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


def main():
    config = load_config()

    results_dir = Path(
        config["paths"]["results"]
    )

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


if __name__ == "__main__":
    main()