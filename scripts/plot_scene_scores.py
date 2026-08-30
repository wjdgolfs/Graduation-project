from pathlib import Path
import csv

import matplotlib.pyplot as plt
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


def load_scores(csv_path):
    frame_numbers = []
    difference_scores = []

    with open(
        csv_path,
        "r",
        encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            frame_numbers.append(
                int(row["frame_number"])
            )

            difference_scores.append(
                float(row["difference_score"])
            )

    return frame_numbers, difference_scores


def plot_scores(
    frame_numbers,
    difference_scores,
    hard_cut_threshold,
    gradual_threshold,
    output_path
):
    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        frame_numbers,
        difference_scores,
        label="Frame Difference"
    )

    plt.axhline(
        hard_cut_threshold,
        linestyle="--",
        label="Hard Cut Threshold"
    )

    plt.axhline(
        gradual_threshold,
        linestyle=":",
        label="Gradual Threshold"
    )

    plt.xlabel(
        "Frame Number"
    )

    plt.ylabel(
        "Difference Score"
    )

    plt.title(
        "Scene Transition Frame Difference"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.show()

    plt.close()


def main():
    config = load_config()

    results_dir = Path(
        config["paths"]["results"]
    )

    csv_path = (
        results_dir
        / "scene_transition_test_scene_scores.csv"
    )

    output_path = (
        results_dir
        / "scene_transition_test_scene_scores.png"
    )

    hard_cut_threshold = (
        config["scene_detection"]["hard_cut_threshold"]
    )

    gradual_threshold = (
        config["scene_detection"]["gradual_threshold"]
    )

    frame_numbers, difference_scores = load_scores(
        csv_path
    )

    plot_scores(
        frame_numbers,
        difference_scores,
        hard_cut_threshold,
        gradual_threshold,
        output_path
    )

    print("-" * 60)
    print("Scene score graph created")
    print("Input:", csv_path)
    print("Output:", output_path)
    print("-" * 60)


if __name__ == "__main__":
    main()