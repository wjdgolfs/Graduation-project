# ============================================================================
# 논문 그림 만들기
# ----------------------------------------------------------------------------
# 지금까지의 결과 파일에서 논문 그림을 만듭니다. 모든 그림은 paths.results/figures 에 PNG(300dpi)로 저장합니다.
#
#   그림 1  전환이 탐지 점수를 흔든다        FTCN 프레임 점수, 대조군 vs 디졸브 64
#   그림 4  전환 구간의 배경·얼굴 변화량      clip_stats.csv
#   그림 5  디졸브 길이별 샷 검출률           shot_detection.csv
#   그림 6  학습된 정규화 강도 α             alpha_frames.npz  ← 제안 모델이 무엇을 배웠는지 보이는 그림
#   그림 7  모델별 전환 강건성                model_comparison.csv
#
#   그림 2  전처리와 신호 계산 파이프라인 (도식)
#   그림 3  제안 모델 구조 (도식)
#
# 실행 예 (프로젝트 폴더에서)
#   python scripts/make_figures.py
#   python scripts/make_figures.py --only 6
# ============================================================================

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 한글이 깨지지 않도록 맑은 고딕을 쓰고, 음수 기호는 유니코드 대신 하이픈을 씁니다.
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
# 로그 눈금 지수(10^-3)는 수식 글꼴로 그려집니다. 맑은 고딕에는 유니코드 음수 기호가 없어 깨지므로 따로 지정합니다.
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

INK = "#222222"
ACCENT = "#C0392B"
MUTED = "#7F8C8D"
BLUE = "#2E5E8E"


def load_config():
    with open(PROJECT_ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


# 로그 눈금 라벨을 10^-3 대신 0.001 처럼 보통 숫자로 찍습니다.
def plain_log_ticks(axes):
    from matplotlib.ticker import FuncFormatter, NullFormatter
    axes.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axes.yaxis.set_minor_formatter(NullFormatter())


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


# 그림 1: 같은 진짜 영상에서 대조군과 디졸브 64 의 FTCN 프레임 점수를 겹쳐 그립니다.
def figure_problem(config, out_dir):
    root = Path(config["benchmark"]["output_dir"]) / "ftcn" / "original"
    center = config["benchmark"]["transition_center"]
    # 전환 때문에 점수가 가장 크게 오른 클립을 고릅니다(대표 예시).
    effect = read_csv(PROJECT_ROOT / "results" / "benchmark" / "ftcn_effect.csv")
    candidates = [row for row in effect
                  if row["source"] == "original" and row["kind"] == "dissolve64" and row["label"] == "0"]
    # 가장 극적인 예가 아니라 효과가 중앙값인 클립을 고릅니다(대표성 있는 예시).
    candidates.sort(key=lambda row: number(row["delta_logit"]))
    clip_id = candidates[len(candidates) // 2]["clip_id"]
    base = clip_id.split("__")[0]
    # 뒤에 이어 붙인 영상의 번호를 manifest 에서 찾습니다.
    manifest = read_csv(Path(config["benchmark"]["output_dir"]) / "manifest.csv")
    partner = next((row["partner_id"] for row in manifest if row["clip_id"] == clip_id), "")

    def scores(name):
        with np.load(root / f"{name}.npz") as data:
            return np.asarray(data["frame_scores"], dtype=np.float64)

    control, treated = scores(f"{base}__control"), scores(clip_id)
    frames = np.arange(len(control))
    partner_path = root / f"{partner}__control.npz"
    partner_scores = scores(f"{partner}__control") if partner and partner_path.exists() else None

    figure, axes = plt.subplots(figsize=(7.2, 3.2))
    axes.axvspan(center - 32, center + 32, color=ACCENT, alpha=0.10, lw=0,
                 label="디졸브 구간 (64프레임)")
    axes.plot(frames, control, color=MUTED, lw=1.4, label=f"앞 영상 {base} 대조군")
    if partner_scores is not None:
        axes.plot(frames, partner_scores, color=MUTED, lw=1.2, ls=":", label=f"뒤 영상 {partner} 대조군")
    axes.plot(frames, treated, color=ACCENT, lw=1.6, label="디졸브 합성본")
    axes.axhline(0.002584857167676091, color=INK, lw=0.9, ls="--", label="판정 임계값")
    axes.set_yscale("log")
    plain_log_ticks(axes)
    axes.set_xlabel("프레임 번호")
    axes.set_ylabel("FTCN 가짜 점수 (로그 눈금)", labelpad=8)
    axes.set_title("진짜 영상 둘을 디졸브로 이으면 그 구간에서 점수가 치솟는다", fontsize=11)
    axes.legend(fontsize=8, loc="upper left", framealpha=0.9)
    figure.savefig(out_dir / "figure1_problem.png")
    plt.close(figure)
    return "figure1_problem.png"


# 그림 4: 전환 구간 안/밖 변화량의 상승배수를 배경과 얼굴로 나눠 막대로 그립니다.
def figure_signal(config, out_dir):
    rows = read_csv(PROJECT_ROOT / "results" / "benchmark" / "clip_stats.csv")
    order = ["control", "hard_cut", "fade8", "fade16", "fade32",
             "dissolve8", "dissolve16", "dissolve32", "dissolve64"]
    labels = ["없음", "하드컷", "페이드 8", "페이드 16", "페이드 32",
              "디졸브 8", "디졸브 16", "디졸브 32", "디졸브 64"]

    def ratio(kind, prefix):
        values = []
        for row in rows:
            name = row["kind"] + (row.get("length") or "")
            if name != kind:
                continue
            inside, outside = number(row[f"{prefix}_in"]), number(row[f"{prefix}_out"])
            if np.isfinite(inside) and np.isfinite(outside) and outside > 0:
                values.append(inside / outside)
        return float(np.median(values)) if values else float("nan")

    background = [ratio(kind, "bg") for kind in order]
    face = [ratio(kind, "face") for kind in order]

    figure, axes = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(order))
    axes.bar(x - 0.2, background, width=0.38, color=BLUE, label="배경 D_bg")
    axes.bar(x + 0.2, face, width=0.38, color=ACCENT, label="얼굴 D_face")
    axes.set_yscale("log")
    plain_log_ticks(axes)
    axes.set_xticks(x)
    axes.set_xticklabels(labels, fontsize=8)
    axes.set_ylabel("전환 구간 안 ÷ 밖 (배)", labelpad=8)
    axes.axhline(1.0, color=INK, lw=0.8)
    axes.set_title("전환은 배경을 얼굴보다 크게 흔든다", fontsize=11)
    axes.legend(fontsize=8)
    figure.savefig(out_dir / "figure4_signal.png")
    plt.close(figure)
    return "figure4_signal.png"


# 그림 5: 디졸브 길이별 샷 검출률을 임계값 3개로 그립니다.
# CSV 의 content* 열에는 검출된 경계의 프레임 번호가 공백으로 구분되어 들어 있습니다(없으면 빈 칸).
# 전환 구간 앞뒤 tolerance 프레임 안에 경계가 하나라도 있으면 검출로 봅니다(detect_shots_benchmark.py 와 같은 규칙).
def detected(row, column, tolerance=8):
    values = (row.get(column) or "").split()
    if not values:
        return False
    start, end = int(row["window_start"]), int(row["window_end"])
    if row["kind"] == "control":
        return True          # 전환이 없는데 경계를 찾았으면 오검출
    return any(start - tolerance <= int(value) <= end + tolerance for value in values)


def figure_shot(config, out_dir):
    rows = read_csv(PROJECT_ROOT / "results" / "benchmark" / "shot_detection.csv")
    lengths = ["8", "16", "32", "64"]
    columns = [("content27", "기본 임계값 27", INK, "o-"),
               ("content15", "민감 15", BLUE, "s--"),
               ("content8", "매우 민감 8", ACCENT, "^:")]

    figure, axes = plt.subplots(figsize=(6.4, 3.4))
    for column, label, color, style in columns:
        rates = []
        for length in lengths:
            group = [row for row in rows if row["kind"] == "dissolve" and row["length"] == length]
            rates.append(100.0 * np.mean([detected(row, column) for row in group]) if group else float("nan"))
        axes.plot(lengths, rates, style, color=color, label=label, lw=1.6, ms=5)

    # 전용 신경망(TransNetV2) 결과가 있으면 함께 그립니다.
    transnet_path = Path(config["paths"]["results"]) / "benchmark" / "shot_detection_transnet.csv"
    if transnet_path.exists():
        transnet = read_csv(transnet_path)
        rates = []
        for length in lengths:
            group = [row for row in transnet if row["kind"] == "dissolve" and row["length"] == length]
            rates.append(100.0 * np.mean([int(row["detected"]) for row in group]) if group else float("nan"))
        axes.plot(lengths, rates, "D-", color="#1E8449", label="TransNetV2 (전용 신경망)", lw=1.8, ms=5)

    controls = [row for row in rows if row["kind"] == "control"]
    false_rate = 100.0 * np.mean([detected(row, "content8") for row in controls])
    axes.axhline(false_rate, color=ACCENT, lw=0.9, ls="-.",
                 label=f"임계값 8 의 오검출 {false_rate:.0f}%")

    axes.set_xlabel("디졸브 길이 (프레임)")
    axes.set_ylabel("검출률 (%)", labelpad=8)
    axes.set_ylim(-3, 103)
    axes.set_title("가장 긴 디졸브는 전용 신경망도 찾지 못한다", fontsize=11)
    axes.legend(fontsize=8)
    figure.savefig(out_dir / "figure5_shot.png")
    plt.close(figure)
    return "figure5_shot.png"


# 그림 6: 학습된 정규화 강도 α. (a) 시간축, (b) 전환 종류별 평균
def figure_alpha(config, out_dir):
    path = Path(config["paths"]["results"]) / "benchmark" / "alpha_frames.npz"
    with np.load(path, allow_pickle=True) as data:
        alpha, kind, frames = data["alpha"], data["kind"], data["frames"]
        window_start, window_end = data["window_start"], data["window_end"]

    figure, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.4),
                                         gridspec_kw={"width_ratios": [1.35, 1]})

    # (a) 짧은 전환만 그립니다. 길이가 30프레임을 넘으면 입력 전체가 전환 구간이라 안/밖 비교가 안 됩니다.
    shown = [("control", "대조군", MUTED, "-"),
             ("dissolve16", "디졸브 16", ACCENT, "-"),
             ("fade16", "페이드 16", BLUE, "--")]
    for name, label, color, style in shown:
        mask = kind == name
        if not mask.any():
            continue
        left.plot(frames, alpha[mask].mean(axis=0), style, color=color, lw=1.8, label=label)
    mask = kind == "dissolve16"
    left.axvspan(window_start[mask][0], window_end[mask][0], color=ACCENT, alpha=0.10, lw=0)
    left.set_xlabel("프레임 번호 (전환 중심 128)")
    left.set_ylabel("정규화 강도 α", labelpad=8)
    left.set_title("(a) 전환 구간에서 α 가 올라간다", fontsize=10)
    left.legend(fontsize=8)

    # (b) 전환 종류별 평균 α
    order = ["control", "hard_cut", "fade8", "fade16", "fade32",
             "dissolve8", "dissolve16", "dissolve32", "dissolve64"]
    labels = ["없음", "하드컷", "페이드 8", "페이드 16", "페이드 32",
              "디졸브 8", "디졸브 16", "디졸브 32", "디졸브 64"]
    means = [float(alpha[kind == name].mean()) if (kind == name).any() else float("nan") for name in order]
    colors = [MUTED] + [INK] + [BLUE] * 3 + [ACCENT] * 4
    right.bar(np.arange(len(order)), means, color=colors)
    right.axhline(means[0], color=MUTED, lw=1.0, ls="--", label="대조군 수준")
    right.set_xticks(np.arange(len(order)))
    right.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    right.set_ylim(min(means) - 0.02, max(means) + 0.02)
    right.set_ylabel("평균 α", labelpad=8)
    right.set_title("(b) 디졸브에서 올리고 페이드에서 내린다", fontsize=10)
    right.legend(fontsize=8)

    figure.savefig(out_dir / "figure6_alpha.png")
    plt.close(figure)
    return "figure6_alpha.png"


# 그림 7: 모델별로 대조군과 전환 클립의 균형 정확도를 비교합니다.
def figure_models(config, out_dir):
    rows = read_csv(PROJECT_ROOT / "results" / "benchmark" / "model_comparison.csv")
    runs = [("b0_ffpp", "B0\n(증강 없음)"), ("b1_augment", "B1\n(증강)"), ("p_normalized", "P\n(제안 모델)")]
    kinds = [("control", "전환 없음", MUTED), ("dissolve32", "디졸브 32", BLUE), ("dissolve64", "디졸브 64", ACCENT)]

    figure, axes = plt.subplots(figsize=(6.8, 3.6))
    width = 0.26
    for index, (kind, label, color) in enumerate(kinds):
        values = []
        for run, _ in runs:
            picked = [row for row in rows if row["run"] == run]
            if kind == "control":
                values.append(100 * number(picked[0]["control_balanced"]) if picked else float("nan"))
            else:
                match = [row for row in picked if row["kind"] == kind]
                values.append(100 * number(match[0]["clip_balanced"]) if match else float("nan"))
        offset = (index - 1) * width
        bars = axes.bar(np.arange(len(runs)) + offset, values, width=width, color=color, label=label)
        axes.bar_label(bars, fmt="%.0f", fontsize=7.5, padding=1)

    axes.set_xticks(np.arange(len(runs)))
    axes.set_xticklabels([label for _, label in runs], fontsize=9)
    axes.set_ylabel("균형 정확도 (%)", labelpad=8)
    axes.set_ylim(50, 100)
    axes.set_title("제안 모델만 전환에서 성능이 떨어지지 않는다", fontsize=11)
    axes.legend(fontsize=8, ncol=3, loc="upper center", framealpha=0.95)
    figure.savefig(out_dir / "figure7_models.png")
    plt.close(figure)
    return "figure7_models.png"



# ── 도식 두 장 ──────────────────────────────────────────────────────────────
# 상자 하나를 그립니다. (x, y) 는 왼쪽 아래 모서리입니다.
def box(axes, x, y, width, height, text, color="#FFFFFF", edge=INK, fontsize=8.5, bold=False):
    from matplotlib.patches import FancyBboxPatch
    patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.08",
                           linewidth=1.1, edgecolor=edge, facecolor=color)
    axes.add_patch(patch)
    axes.text(x + width / 2, y + height / 2, text, ha="center", va="center",
              fontsize=fontsize, color=INK, fontweight="bold" if bold else "normal", linespacing=1.4)


def arrow(axes, start, end, text=None, color=INK, style="-|>"):
    from matplotlib.patches import FancyArrowPatch
    axes.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=11,
                                   linewidth=1.1, color=color, shrinkA=2, shrinkB=2))
    if text:
        axes.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.12, text,
                  ha="center", va="bottom", fontsize=7.5, color=color)


def blank_axes(figure_size):
    figure, axes = plt.subplots(figsize=figure_size)
    axes.set_xlim(0, 10)
    axes.set_ylim(0, 10)
    axes.axis("off")
    axes.grid(False)
    return figure, axes


# 그림 2: 전처리와 신호 계산 파이프라인
def figure_pipeline(config, out_dir):
    figure, axes = blank_axes((7.6, 3.2))
    axes.set_ylim(0, 6)

    box(axes, 0.1, 2.3, 1.5, 1.3, "영상\n(1클립)", color="#F2F2F2")
    box(axes, 2.0, 2.3, 1.9, 1.3, "얼굴 검출·추적\nMediaPipe 478점", color="#FFFFFF")
    box(axes, 4.3, 4.0, 2.2, 1.3, "부위 마스크\n눈·입·경계·얼굴", color="#FDECEA", edge=ACCENT)
    box(axes, 4.3, 0.6, 2.2, 1.3, "배경 마스크\n얼굴 상자 2.0배 제외", color="#E8F0F8", edge=BLUE)
    box(axes, 6.9, 4.0, 1.5, 1.3, "D_region", color="#FDECEA", edge=ACCENT, bold=True)
    box(axes, 6.9, 0.6, 1.5, 1.3, "D_bg", color="#E8F0F8", edge=BLUE, bold=True)
    box(axes, 8.7, 2.3, 1.2, 1.3, "R =\nD_region\n- α·D_bg", color="#FFFFFF", fontsize=8, bold=True)

    arrow(axes, (1.6, 2.95), (2.0, 2.95))
    arrow(axes, (3.9, 3.3), (4.3, 4.3))
    arrow(axes, (3.9, 2.6), (4.3, 1.5))
    arrow(axes, (6.5, 4.65), (6.9, 4.65))
    arrow(axes, (6.5, 1.25), (6.9, 1.25))
    arrow(axes, (8.4, 4.4), (9.3, 3.6), color=ACCENT)
    arrow(axes, (8.4, 1.5), (9.3, 2.3), color=BLUE)

    axes.text(5.0, 5.6, "프레임 t-1 과 t 에서 공통인 화소만 비교한다", fontsize=8, color=MUTED, ha="center")
    figure.savefig(out_dir / "figure2_pipeline.png")
    plt.close(figure)
    return "figure2_pipeline.png"


# 그림 3: 제안 모델 구조
def figure_model(config, out_dir):
    figure, axes = blank_axes((7.8, 3.6))
    axes.set_ylim(0, 6.6)

    box(axes, 0.05, 4.1, 1.6, 1.1, "얼굴 크롭\n30×240×240", color="#F2F2F2", fontsize=8)
    box(axes, 1.9, 4.1, 2.2, 1.1, "Xception → CBAM", color="#FDECEA", edge=ACCENT)
    box(axes, 4.4, 4.1, 1.3, 1.1, "F_t", color="#FDECEA", edge=ACCENT, bold=True)

    box(axes, 0.05, 1.0, 1.6, 1.1, "배경 썸네일\n30×96×128", color="#F2F2F2", fontsize=8)
    box(axes, 1.9, 1.0, 2.2, 1.1, "전역 분기\n합성곱 4층", color="#E8F0F8", edge=BLUE)
    box(axes, 4.4, 1.0, 1.3, 1.1, "b_t", color="#E8F0F8", edge=BLUE, bold=True)
    box(axes, 4.25, 2.5, 1.6, 0.9, "게이트 → α_t", color="#FFFFFF", edge=BLUE, fontsize=8)

    box(axes, 6.0, 2.5, 1.5, 1.3, "정규화\nF_t - α_t·g_t", color="#FFFFFF", bold=True, fontsize=8.5)
    box(axes, 7.8, 2.5, 1.5, 1.3, "Bi-ConvLSTM\n→ FC", color="#F2F2F2", fontsize=8.5)
    box(axes, 9.0, 0.6, 0.9, 0.9, "real\n/ fake", color="#FFFFFF", fontsize=8)

    arrow(axes, (1.65, 4.65), (1.9, 4.65))
    arrow(axes, (4.1, 4.65), (4.4, 4.65))
    arrow(axes, (1.65, 1.55), (1.9, 1.55))
    arrow(axes, (4.1, 1.55), (4.4, 1.55))
    arrow(axes, (5.05, 2.1), (5.05, 2.5), color=BLUE)
    arrow(axes, (5.7, 4.4), (6.3, 3.8), color=ACCENT)
    arrow(axes, (5.7, 1.5), (6.3, 2.5), color=BLUE)
    arrow(axes, (5.85, 2.95), (6.0, 2.95), color=BLUE)
    arrow(axes, (7.5, 3.15), (7.8, 3.15))
    arrow(axes, (8.55, 2.5), (9.3, 1.5))

    axes.text(5.0, 6.1, "얼굴 특징에서 전역 성분을 빼되, 얼마나 뺄지(α_t)는 배경의 시간 패턴이 정한다",
              fontsize=8.5, color=MUTED, ha="center")
    axes.text(5.05, 0.55, "게이트 입력: b_t, 시간 차분, 배경 변화량", fontsize=7.5, color=BLUE, ha="center")
    figure.savefig(out_dir / "figure3_model.png")
    plt.close(figure)
    return "figure3_model.png"


FIGURES = {1: figure_problem, 2: figure_pipeline, 3: figure_model, 4: figure_signal, 5: figure_shot, 6: figure_alpha, 7: figure_models}


def main():
    parser = argparse.ArgumentParser(description="논문 그림을 만듭니다.")
    parser.add_argument("--only", type=int, nargs="+", default=sorted(FIGURES), help="만들 그림 번호")
    args = parser.parse_args()

    config = load_config()
    out_dir = Path(config["paths"]["results"]) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for number_ in args.only:
        maker = FIGURES.get(number_)
        if maker is None:
            print(f"그림 {number_} 은 이 스크립트에서 만들지 않습니다(도식).")
            continue
        try:
            name = maker(config, out_dir)
            print(f"그림 {number_}: {out_dir / name}")
        except Exception as error:  # 하나가 실패해도 나머지는 만듭니다.
            print(f"그림 {number_} 실패: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
