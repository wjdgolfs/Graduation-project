# 프레임별 변화 점수를 그래프로 그려 전환 위치와 기준값을 눈으로 비교합니다.
# 먼저 detect_scene_transition.py를 실행해 입력 CSV를 만들어야 합니다.
# 출력: results의 PNG 그래프와 화면에 표시되는 그래프 창.

from pathlib import Path
import csv

import matplotlib.pyplot as plt
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


# CSV 열 이름으로 값을 읽어 그래프의 X축(번호)과 Y축(점수) 목록을 반환합니다.
def load_scores(csv_path):
    frame_numbers = []
    difference_scores = []

    with open(
        csv_path,
        "r",
        encoding="utf-8"
    ) as file:
        # CSV 첫 줄의 열 이름으로 값을 찾습니다. 문자열을 숫자로 변환해야 수치 축에 그릴 수 있습니다.
        reader = csv.DictReader(file)

        for row in reader:
            frame_numbers.append(
                int(row["frame_number"])
            )

            difference_scores.append(
                float(row["difference_score"])
            )

    return frame_numbers, difference_scores


# 점수 곡선과 두 기준선을 그리고 output_path에 PNG를 저장한 뒤 화면에 보여 줍니다.
def plot_scores(
    frame_numbers,
    difference_scores,
    hard_cut_threshold,
    gradual_threshold,
    output_path
):
    # 가로 12, 세로 6인치의 그래프를 만들어 시간에 따른 변화를 보기 쉽게 합니다.
    plt.figure(
        figsize=(12, 6)
    )

    # X축은 원본 프레임 번호, Y축은 바로 이전 프레임과의 평균 밝기 차이입니다.
    plt.plot(
        frame_numbers,
        difference_scores,
        label="Frame Difference"
    )

    # 수평 기준선으로 점수가 설정값을 넘는 위치를 비교합니다. 구간 길이 판정 자체를 그리지는 않습니다.
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

    # 축 제목과 범례가 그림 경계에 잘리지 않도록 여백을 자동 조정합니다.
    plt.tight_layout()

    # 화면 창을 닫아도 볼 수 있도록 200 DPI 해상도의 PNG 파일로 먼저 저장합니다.
    plt.savefig(
        output_path,
        dpi=200
    )

    # 일반적인 데스크톱 실행에서는 그래프 창을 닫을 때까지 여기서 기다립니다.
    plt.show()

    # 그래프에 사용한 자원을 정리합니다.
    plt.close()


# 이 파일을 직접 실행할 때 사용할 입력 경로와 설정을 준비하고 작업을 시작합니다.
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


# 다른 파일에서 import할 때는 실행하지 않고, 이 파일을 직접 실행했을 때만 아래 작업을 시작합니다.
if __name__ == "__main__":
    main()
