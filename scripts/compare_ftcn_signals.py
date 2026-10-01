# ============================================================================
# 3→5단계 연결: 신호에서 잰 전환 크기가 탐지기 점수 변화를 설명하는가
# ----------------------------------------------------------------------------
# 묻는 것
#   analyze_benchmark.py 는 "전환이 배경(D_bg)을 얼굴(D_face)보다 3~4배 크게 흔든다"를 보여 주고,
#   analyze_benchmark_ftcn.py 는 "전환이 FTCN 점수를 real 에서 올린다(오탐)"를 보여 줍니다.
#   이 스크립트는 둘을 클립 단위로 붙여서, 배경이 더 크게 흔들린 클립이 실제로 더 크게 속는지 봅니다.
#
# 왜 길이별로 나누는가
#   디졸브가 길수록 프레임당 변화는 작아지는데(배경 상승배수 4.2배 → 2.7배) 점수 변화는 커집니다(+2.0 → +3.3).
#   길이를 섞어 상관을 구하면 이 두 방향이 서로를 지워 0 에 가까운 값이 나옵니다.
#
# 보는 값 (전환 구간 안 ÷ 바깥)
#   배경 상승배수      D_bg(안) / D_bg(밖)
#   얼굴 상승배수      D_face(안) / D_face(밖)
#   정규화 잔차 상승배수  R = D_face − α·D_bg 의 안/밖 비율 (α 는 analyze_benchmark.py 가 구한 얼굴 계수)
#
# 입력: paths.results/benchmark/clip_stats.csv (analyze_benchmark.py)
#       paths.results/benchmark/ftcn_effect.csv (analyze_benchmark_ftcn.py)
# 출력: 화면 표 + paths.results/benchmark/ftcn_vs_signals.csv (클립마다 붙인 값)
#
# 실행 예 (프로젝트 폴더에서)
#   python scripts/compare_ftcn_signals.py
# ============================================================================

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# analyze_benchmark.py 가 최소제곱으로 구한 얼굴 부위의 배경 계수입니다(3단계 결과).
DEFAULT_ALPHA = 1.27

MEASURES = (
    ("bg_jump", "배경 상승배수"),
    ("face_jump", "얼굴 상승배수"),
    ("resid_jump", "정규화 잔차 상승배수"),
)


def load_config():
    with open(PROJECT_ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# 빈 칸과 "nan" 을 nan 으로 읽습니다.
def number(row, key):
    value = row.get(key, "")
    return float(value) if value not in ("", "nan", None) else float("nan")


# 두 CSV 를 (source, clip_id) 로 붙입니다. 둘 다 analyze_* 스크립트가 같은 키로 씁니다.
def join_rows(stats_path, effect_path, alpha):
    stats = {(row["source"], row["clip_id"]): row
             for row in csv.DictReader(open(stats_path, newline="", encoding="utf-8"))}

    joined = []
    for row in csv.DictReader(open(effect_path, newline="", encoding="utf-8")):
        signal = stats.get((row["source"], row["clip_id"]))
        if signal is None:
            continue
        bg_in, bg_out = number(signal, "bg_in"), number(signal, "bg_out")
        face_in, face_out = number(signal, "face_in"), number(signal, "face_out")
        resid_in, resid_out = face_in - alpha * bg_in, face_out - alpha * bg_out
        joined.append({
            "clip_id": row["clip_id"],
            "source": row["source"],
            "label": int(row["label"]),
            "kind": row["kind"],
            "delta_logit": float(row["delta_logit"]),
            # 바깥 값이 0 이하면 비율이 뜻을 잃으므로 nan 으로 둡니다.
            "bg_jump": bg_in / bg_out if bg_out > 0 else float("nan"),
            "face_jump": face_in / face_out if face_out > 0 else float("nan"),
            "resid_jump": resid_in / resid_out if abs(resid_out) > 1e-9 else float("nan"),
        })
    return joined


# 한 묶음(정답 × 전환 종류)에서 측정값과 점수 변화의 순위상관을 구합니다.
def correlate(group, field):
    values = np.array([row[field] for row in group], dtype=np.float64)
    deltas = np.array([row["delta_logit"] for row in group], dtype=np.float64)
    usable = np.isfinite(values) & np.isfinite(deltas)
    if usable.sum() < 6:
        return float("nan"), float("nan"), float("nan"), int(usable.sum())
    rho, p = spearmanr(values[usable], deltas[usable])
    return float(np.median(values[usable])), float(rho), float(p), int(usable.sum())


def main():
    parser = argparse.ArgumentParser(description="신호에서 잰 전환 크기와 FTCN 점수 변화를 클립 단위로 비교합니다.")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="정규화 잔차에 쓸 배경 계수")
    args = parser.parse_args()

    config = load_config()
    results = Path(config["paths"]["results"]) / "benchmark"
    rows = join_rows(results / "clip_stats.csv", results / "ftcn_effect.csv", args.alpha)
    if not rows:
        print("붙일 클립이 없습니다. analyze_benchmark.py 와 analyze_benchmark_ftcn.py 를 먼저 돌리세요.")
        return

    print(f"두 결과를 붙인 클립 {len(rows)}개 (α = {args.alpha})\n")
    print(f"{'정답':<6}{'전환':<12}{'n':>4}{'logit 변화':>12}   " + "".join(f"{label:>26}" for _, label in MEASURES))

    kinds = sorted({row["kind"] for row in rows})
    for label_value, label_name in ((0, "real"), (1, "fake")):
        for kind in kinds:
            group = [row for row in rows if row["label"] == label_value and row["kind"] == kind]
            if len(group) < 6:
                continue
            deltas = np.array([row["delta_logit"] for row in group])
            line = f"{label_name:<6}{kind:<12}{len(group):>4}{np.median(deltas):>+12.2f}   "
            for field, _ in MEASURES:
                median, rho, p, _ = correlate(group, field)
                line += f"{median:>8.2f}배 rho{rho:>+6.2f} p{p:>5.2f}"
            print(line)

    out_path = results / "ftcn_vs_signals.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n클립별 값 저장: {out_path}")


if __name__ == "__main__":
    main()
