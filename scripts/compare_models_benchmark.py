# ============================================================================
# 5단계 평가: 전환 구간에서 어느 모델이 덜 흔들리는가
# ----------------------------------------------------------------------------
# 묻는 것
#   scripts/score_benchmark_paper.py 가 모델마다 남긴 effect_<이름>.csv 를 모아
#   "전환이 없을 때"와 "전환이 있을 때"의 성능을 모델끼리 비교합니다.
#
# 왜 균형 정확도를 쓰는가
#   전환은 real 점수를 올리고(오탐) fake 점수를 내립니다(미탐). 한쪽만 보면 모델을 잘못 고릅니다.
#   예를 들어 모든 점수를 낮추는 모델은 오탐이 줄지만 미탐이 늘어납니다.
#   그래서 real 을 맞힌 비율과 fake 를 맞힌 비율의 평균(균형 정확도)으로 함께 봅니다.
#
# 입력: paths.results/benchmark/effect_<이름>.csv (모델마다 하나)
# 출력: 화면 표 + paths.results/benchmark/model_comparison.csv
#
# 실행 예 (프로젝트 폴더에서)
#   python scripts/compare_models_benchmark.py --runs b0_ffpp b1_augment p_normalized
# ============================================================================

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DECISION_THRESHOLD = 0.5


def load_config():
    with open(PROJECT_ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# effect_*.csv 한 개를 읽습니다. 한 줄이 "대조군과 짝지은 전환 클립" 하나입니다.
def load_rows(path):
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["label"] = int(row["label"])
        for key in ("control_score", "score", "delta_logit", "alpha", "control_alpha", "delta_alpha"):
            row[key] = float(row[key]) if row.get(key) not in ("", None) else float("nan")
    return rows


def rate(values):
    return float(np.mean(values)) if len(values) else float("nan")


# 유한한 값만 평균냅니다. 하나도 없으면 nan 입니다(제안 모델이 아닌 경우의 α).
def mean_or_nan(values):
    usable = [value for value in values if np.isfinite(value)]
    return float(np.mean(usable)) if usable else float("nan")


# 한 묶음(전환 종류)에서 대조군과 전환 클립의 맞힌 비율을 구합니다.
#   real 은 점수가 기준 미만이어야 맞힌 것이고, fake 는 기준 이상이어야 맞힌 것입니다.
def summarize(rows, kind):
    group = [row for row in rows if row["kind"] == kind]
    real = [row for row in group if row["label"] == 0]
    fake = [row for row in group if row["label"] == 1]

    control_real = rate([row["control_score"] < DECISION_THRESHOLD for row in real])
    control_fake = rate([row["control_score"] >= DECISION_THRESHOLD for row in fake])
    clip_real = rate([row["score"] < DECISION_THRESHOLD for row in real])
    clip_fake = rate([row["score"] >= DECISION_THRESHOLD for row in fake])

    return {
        "kind": kind,
        "pairs": len(group),
        "control_real_ok": control_real,
        "control_fake_ok": control_fake,
        "control_balanced": (control_real + control_fake) / 2,
        "clip_real_ok": clip_real,
        "clip_fake_ok": clip_fake,
        "clip_balanced": (clip_real + clip_fake) / 2,
        "drop": (clip_real + clip_fake) / 2 - (control_real + control_fake) / 2,
        "real_delta_logit": float(np.median([row["delta_logit"] for row in real])),
        "fake_delta_logit": float(np.median([row["delta_logit"] for row in fake])),
        # α 는 제안 모델에만 있습니다. 나머지 모델에서는 전부 nan 이므로 평균을 구하지 않습니다.
        "alpha_shift": mean_or_nan([row["delta_alpha"] for row in group]),
    }


def main():
    config = load_config()
    results = Path(config["paths"]["results"]) / "benchmark"

    parser = argparse.ArgumentParser(description="모델별 전환 강건성을 비교합니다.")
    parser.add_argument("--runs", nargs="+", default=["b0_ffpp", "b1_augment", "p_normalized"],
                        help="비교할 학습 이름 (effect_<이름>.csv 가 있어야 합니다)")
    args = parser.parse_args()

    records = []
    for run in args.runs:
        path = results / f"effect_{run}.csv"
        if not path.exists():
            print(f"건너뜀: {path.name} 이 없습니다. score_benchmark_paper.py 를 먼저 돌리세요.")
            continue
        rows = load_rows(path)
        for kind in sorted({row["kind"] for row in rows}):
            records.append({"run": run, **summarize(rows, kind)})

    if not records:
        return

    print("\n전환이 없을 때(대조군)와 있을 때의 맞힌 비율. 균형 정확도 = (real 맞힘 + fake 맞힘) / 2\n")
    print(f"{'모델':<14}{'전환':<12}{'쌍':>4}{'대조군 균형':>12}{'전환 균형':>11}{'변화':>8}"
          f"{'real 오탐':>18}{'fake 탐지':>16}")
    for record in records:
        print(f"{record['run']:<14}{record['kind']:<12}{record['pairs']:>4}"
              f"{record['control_balanced']:>11.1%}{record['clip_balanced']:>11.1%}"
              f"{record['drop']:>+8.1%}"
              f"{1 - record['control_real_ok']:>10.0%} → {1 - record['clip_real_ok']:<5.0%}"
              f"{record['control_fake_ok']:>8.0%} → {record['clip_fake_ok']:<5.0%}")

    print(f"\n{'모델':<14}{'전환':<12}{'real logit 변화':>17}{'fake logit 변화':>17}{'α 변화':>10}")
    for record in records:
        alpha = f"{record['alpha_shift']:+.3f}" if np.isfinite(record["alpha_shift"]) else "—"
        print(f"{record['run']:<14}{record['kind']:<12}{record['real_delta_logit']:>+17.3f}"
              f"{record['fake_delta_logit']:>+17.3f}{alpha:>10}")

    out_path = results / "model_comparison.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
