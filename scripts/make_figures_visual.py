# ============================================================================
# 논문용 시각 자료 만들기 (실제 영상 프레임이 들어가는 그림)
# ----------------------------------------------------------------------------
# 원 논문[1]의 Fig. 9(Grad-CAM)처럼 "무엇을 보고 있는지"를 눈으로 보이는 그림들입니다.
#
#   A  전환 예시       벤치마크 클립에서 하드컷·페이드·디졸브가 실제로 어떻게 보이는지
#   B  부위·배경 마스크  눈·입·경계·얼굴 마스크와 배경 마스크가 얼굴 위에 어떻게 그려지는지
#   C  혼동 행렬        모델별 TP/FP/TN/FN (원 논문 Fig. 8 과 같은 형식)
#
# 실행 예 (프로젝트 폴더에서)
#   python scripts/make_figures_visual.py
#   python scripts/make_figures_visual.py --only A
# ============================================================================

import argparse
import csv
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.face_regions import background_mask, region_point_indices  # noqa: E402
from preprocessing.region_signals import region_masks  # noqa: E402

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

INK = "#222222"
ACCENT = "#C0392B"
BLUE = "#2E5E8E"


def load_config():
    with open(PROJECT_ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def read_frames(path, numbers):
    capture = cv2.VideoCapture(str(path))
    frames, index, wanted = {}, 0, set(numbers)
    while wanted:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            frames[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            wanted.discard(index)
        index += 1
    capture.release()
    return frames


# 그림 A: 전환 세 종류가 실제로 어떻게 보이는지 프레임 띠로 보여 줍니다.
def figure_transition_examples(config, out_dir):
    benchmark = Path(config["benchmark"]["output_dir"])
    with open(benchmark / "manifest.csv", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    # 같은 기준 영상에서 만든 세 전환을 고릅니다(비교가 쉽도록).
    base = next(row["base_id"] for row in rows
                if row["source"] == "original" and row["kind"] == "dissolve" and row["length"] == "64")
    picks = [("hard_cut", "0", "하드컷"), ("fade", "32", "페이드 32프레임"), ("dissolve", "64", "디졸브 64프레임")]
    numbers = [96, 112, 128, 144, 160]

    figure, axes = plt.subplots(len(picks), len(numbers), figsize=(9.0, 5.4))
    for row_index, (kind, length, label) in enumerate(picks):
        row = next(r for r in rows if r["source"] == "original" and r["base_id"] == base
                   and r["kind"] == kind and r["length"] == length)
        frames = read_frames(row["path"], numbers)
        start, end = int(row["window_start"]), int(row["window_end"])
        for column, number in enumerate(numbers):
            axis = axes[row_index, column]
            axis.imshow(frames.get(number, np.zeros((10, 10, 3), dtype=np.uint8)))
            axis.set_xticks([])
            axis.set_yticks([])
            inside = start <= number < end or (kind == "hard_cut" and number == 128)
            for spine in axis.spines.values():
                spine.set_edgecolor(ACCENT if inside else "#CCCCCC")
                spine.set_linewidth(2.0 if inside else 0.8)
            if row_index == 0:
                axis.set_title(f"{number}번", fontsize=9)
            if column == 0:
                # 세로로 눕히면 읽기 어려워 가로로 둡니다.
                axis.set_ylabel(label, fontsize=9.5, rotation=0, ha="right", va="center", labelpad=10)
    figure.suptitle("전환 구간의 실제 프레임 (붉은 테두리가 전환 구간)", fontsize=11, y=0.98)
    figure.savefig(out_dir / "figureA_transition_examples.png")
    plt.close(figure)
    return "figureA_transition_examples.png"


# 그림 B: 부위 마스크와 배경 마스크를 실제 얼굴 위에 그립니다.
def figure_masks(config, out_dir):
    faces_root = Path(config["preprocess"]["output_dir"]) / "ffpp" / "original"
    path = sorted(faces_root.glob("*.npz"))[0]
    with np.load(path) as data:
        face = data["faces"][10]
        landmarks = data["landmarks"][9:12]
        thumb = data["thumbs"][10]
        box = data["smoothed_boxes"][10]
        scale = float(data["thumb_scale"])

    settings = dict(config["regions"])
    # 부위 마스크는 평활한 랜드마크로 만듭니다(2단계와 같은 처리). 가운데 프레임만 씁니다.
    from preprocessing.face_regions import smooth_landmarks
    points = smooth_landmarks(landmarks, settings["landmark_smoothing"])[1]
    masks = region_masks(points, region_point_indices(), face.shape[0], settings)
    keys = ["eyes", "mouth", "boundary", "face"]
    names = ["눈", "입", "얼굴 경계", "얼굴 전체"]
    colors = [(230, 80, 60), (60, 140, 220), (240, 180, 40), (120, 200, 120)]

    figure, axes = plt.subplots(1, 6, figsize=(11.0, 2.3))
    axes[0].imshow(face)
    axes[0].set_title("얼굴 크롭", fontsize=9)
    for index, (name, color) in enumerate(zip(names, colors)):
        overlay = face.copy()
        mask = masks[keys[index]]
        overlay[mask] = (0.45 * np.array(color) + 0.55 * overlay[mask]).astype(np.uint8)
        axes[index + 1].imshow(overlay)
        axes[index + 1].set_title(name, fontsize=9)

    exclusion = background_mask(thumb.shape[0], thumb.shape[1],
                                np.asarray(box) * scale, config["preprocess"]["background_exclusion_scale"])
    shown = cv2.cvtColor(thumb, cv2.COLOR_GRAY2RGB)
    shown[~exclusion] = (shown[~exclusion] * 0.25).astype(np.uint8)
    axes[5].imshow(shown)
    axes[5].set_title("배경 (얼굴 제외)", fontsize=9)

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("부위 마스크와 배경 마스크", fontsize=11, y=1.04)
    figure.savefig(out_dir / "figureB_masks.png")
    plt.close(figure)
    return "figureB_masks.png"


# 그림 C: 모델별 혼동 행렬 (원 논문 Fig. 8 과 같은 형식)
def figure_confusion(config, out_dir):
    runs = [("B0 (증강 없음)", 99, 43, 41, 517), ("B1 (전환 증강)", 105, 39, 35, 521),
            ("P (제안 모델)", 90, 41, 50, 519)]
    figure, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    for axis, (name, tn, fn, fp, tp) in zip(axes, runs):
        matrix = np.array([[tn, fp], [fn, tp]], dtype=float)
        shown = matrix / matrix.sum(axis=1, keepdims=True)
        axis.imshow(shown, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                axis.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center",
                          fontsize=12, color="white" if shown[i, j] > 0.5 else INK, fontweight="bold")
        axis.set_xticks([0, 1]); axis.set_xticklabels(["정상 판정", "가짜 판정"], fontsize=8.5)
        axis.set_yticks([0, 1]); axis.set_yticklabels(["정상", "가짜"], fontsize=8.5)
        axis.set_title(name, fontsize=10)
        axis.set_xlabel("예측", fontsize=9)
    axes[0].set_ylabel("실제", fontsize=9)
    figure.suptitle("전환이 없는 테스트 집합의 혼동 행렬 (정상 140 + 가짜 560)", fontsize=11, y=1.03)
    figure.savefig(out_dir / "figureC_confusion.png")
    plt.close(figure)
    return "figureC_confusion.png"



# 그림 D: 같은 장면의 진짜 영상과 딥페이크 영상, 그리고 바뀐 자리
def figure_real_vs_fake(config, out_dir):
    raw = Path(config["paths"]["raw"]) / "FaceForensics++"
    real_path = raw / "original_sequences" / "youtube" / "c23" / "videos" / "000.mp4"
    fakes = [("Deepfakes", "Deepfakes"), ("Face2Face", "Face2Face"), ("FaceSwap", "FaceSwap"),
             ("NeuralTextures", "NeuralTextures")]
    numbers = [40]

    real = read_frames(real_path, numbers)[numbers[0]]
    figure, axes = plt.subplots(2, 1 + len(fakes), figsize=(11.5, 4.6))

    axes[0, 0].imshow(real)
    axes[0, 0].set_title("원본 (진짜)", fontsize=10, color=BLUE)
    axes[1, 0].axis("off")

    for index, (folder, label) in enumerate(fakes, start=1):
        path = raw / "manipulated_sequences" / folder / "c23" / "videos" / "000_003.mp4"
        fake = read_frames(path, numbers)[numbers[0]]
        if fake.shape != real.shape:
            fake = cv2.resize(fake, (real.shape[1], real.shape[0]))
        axes[0, index].imshow(fake)
        axes[0, index].set_title(label, fontsize=10, color=ACCENT)

        # 원본과의 차이: 조작이 어디에 있는지 보여 줍니다.
        difference = np.abs(fake.astype(np.int16) - real.astype(np.int16)).max(axis=2)
        axes[1, index].imshow(difference, cmap="inferno", vmin=0, vmax=max(40, difference.max() * 0.6))
        axes[1, index].set_title("원본과의 차이", fontsize=8.5)

    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    axes[0, 0].set_ylabel("프레임", fontsize=9, rotation=0, ha="right", va="center", labelpad=10)
    figure.suptitle("같은 장면의 원본과 네 가지 딥페이크. 아래는 원본과 달라진 자리 (밝을수록 큰 차이)",
                    fontsize=11, y=0.98)
    figure.savefig(out_dir / "figureD_real_vs_fake.png")
    plt.close(figure)
    return "figureD_real_vs_fake.png"


MAKERS = {"A": figure_transition_examples, "D": figure_real_vs_fake, "B": figure_masks, "C": figure_confusion}


def main():
    parser = argparse.ArgumentParser(description="실제 영상이 들어가는 논문 그림을 만듭니다.")
    parser.add_argument("--only", nargs="+", default=sorted(MAKERS), help="만들 그림 (A, B, C)")
    args = parser.parse_args()

    config = load_config()
    out_dir = Path(config["paths"]["results"]) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in args.only:
        maker = MAKERS.get(key.upper())
        if maker is None:
            print(f"그림 {key} 는 없습니다.")
            continue
        try:
            print(f"그림 {key.upper()}: {out_dir / maker(config, out_dir)}")
        except Exception as error:
            print(f"그림 {key.upper()} 실패: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
