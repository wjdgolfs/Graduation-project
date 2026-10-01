# ============================================================================
# 3단계 벤치마크 분석: 배경 정규화가 장면 전환 효과를 지우는가
# ----------------------------------------------------------------------------
# 묻는 것 세 가지
#   1) 전환은 부위 변화량(D_region)과 배경 변화량(D_bg)을 각각 몇 배로 키우는가
#   2) R = D_region - α·D_bg 로 정규화하면 전환 때문에 생긴 증가가 사라지는가
#   3) 정규화한 뒤에도 real 과 fake 의 차이는 남는가 (전환 효과만 지우고 신호는 지키는지)
#
# 구간 나누기
#   전환 구간 안 : manifest 의 window_start ~ window_end (하드컷은 전환 지점 앞뒤 1프레임)
#   전환 구간 밖 : 그 바깥. 단 경계에서 margin(4) 프레임 이내는 어느 쪽에도 넣지 않습니다.
#                  압축과 블러 때문에 변화가 경계 너머로 조금 번지기 때문입니다.
#   control(전환 없음) 클립은 중심 프레임 앞뒤 1프레임을 "안"으로 봐서, 전환이 없을 때의 기준값을 줍니다.
#
# α 를 정하는 방법
#   전환이 만든 "증가분"끼리 맞춥니다. 클립마다
#       Δ부위 = (구간 안 평균) - (구간 밖 평균),   Δ배경 = (구간 안 평균) - (구간 밖 평균)
#   을 구하고, 전환 클립 전체에서 Δ부위 ≈ α · Δ배경 이 되도록 최소제곱으로 α 를 구합니다.
#   즉 "배경이 이만큼 흔들릴 때 부위는 평균 이만큼 흔들린다"는 비율입니다.
#   control 클립은 전환이 없으므로 α 추정에서 뺍니다.
#
# 읽는 법
#   - "배수"가 1 에 가까우면 그 지표는 전환에 흔들리지 않는다는 뜻입니다.
#   - 정규화 전 배수가 크고, 정규화 뒤 "남은 증가분"이 0 에 가까우면 정규화가 통한 것입니다.
#   - 마지막 표에서 정규화 전후의 AUC 가 비슷하면, 전환 효과만 지우고 real/fake 신호는 지킨 것입니다.
#
# 입력: benchmark.output_dir/signals/**.npz (preprocessing/benchmark_signals.py 결과)
# 출력: 화면 표 + paths.results/benchmark/clip_stats.csv (클립마다 구간 안/밖 평균값)
#
# 실행 예 (프로젝트 폴더에서)
#   python scripts/analyze_benchmark.py
# ============================================================================

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import mannwhitneyu


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

REGIONS = ["eyes", "mouth", "boundary", "face"]
# 전환 구간 경계에서 이만큼 떨어진 프레임만 "바깥"으로 셉니다.
MARGIN = 4
# 표에 나오는 전환 순서
KIND_ORDER = ["control", "hard_cut", "fade8", "fade16", "fade32", "dissolve8", "dissolve16", "dissolve32", "dissolve64"]


def load_config():
    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return yaml.safe_load(file)


# 얼굴을 못 찾은 프레임 등으로 쓸 값이 하나도 없으면 nan 입니다(np.nanmean 은 이때 경고를 냅니다).
def mean_or_nan(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


# 클립 하나에서 전환 구간 안/밖의 평균 변화량을 구합니다. 계산할 수 없으면 None 입니다.
def clip_stats(path, center):
    with np.load(path) as data:
        if str(data["error"]).strip():
            return None
        diff = data["region_diff"].astype(float)
        background = data["background_diff"].astype(float)
        frames = data["frame_indices"]
        kind = str(data["kind"])
        length = int(data["length"])
        start, end = int(data["window_start"]), int(data["window_end"])
        record = {
            "clip_id": str(data["clip_id"]),
            "source": str(data["source"]),
            "label": int(data["label"]),
            "kind": kind if not length else f"{kind}{length}",
        }

    if kind == "control":
        # 전환이 없는 클립은 같은 위치(중심 앞뒤 1프레임)를 "안"으로 봐서 기준값을 만듭니다.
        inside = np.abs(frames - center) <= 1
        near = np.abs(frames - center) <= MARGIN
    elif start == end:
        # 하드컷은 구간 길이가 0 이라 전환 지점 앞뒤 1프레임을 봅니다.
        inside = np.abs(frames - start) <= 1
        near = np.abs(frames - start) <= MARGIN
    else:
        inside = (frames >= start) & (frames < end)
        near = (frames >= start - MARGIN) & (frames < end + MARGIN)

    outside = ~near
    outside[0] = False  # 첫 프레임은 비교할 앞 프레임이 없어 nan 입니다.
    if inside.sum() < 1 or outside.sum() < 10:
        return None

    record["bg_in"] = mean_or_nan(background[inside])
    record["bg_out"] = mean_or_nan(background[outside])
    for index, name in enumerate(REGIONS):
        record[f"{name}_in"] = mean_or_nan(diff[inside, index])
        record[f"{name}_out"] = mean_or_nan(diff[outside, index])
    return record


# 안/밖 비율의 중앙값. 바깥 값이 0 이면 뺍니다.
def ratio_median(records, name):
    values = [record[f"{name}_in"] / record[f"{name}_out"] for record in records
              if record[f"{name}_out"] > 0 and np.isfinite(record[f"{name}_in"])]
    return float(np.median(values)) if values else float("nan")


# Δ부위 ≈ α · Δ배경 이 되도록 최소제곱으로 기울기를 구합니다(절편 없는 직선).
def fit_alpha(records, name):
    delta_bg = np.array([record["bg_in"] - record["bg_out"] for record in records])
    delta_region = np.array([record[f"{name}_in"] - record[f"{name}_out"] for record in records])
    ok = np.isfinite(delta_bg) & np.isfinite(delta_region)
    if ok.sum() < 10 or not np.any(delta_bg[ok] > 0):
        return float("nan")
    return float((delta_bg[ok] * delta_region[ok]).sum() / (delta_bg[ok] ** 2).sum())


def auc_of(records, score_of):
    real = np.array([score_of(record) for record in records if record["label"] == 0])
    fake = np.array([score_of(record) for record in records if record["label"] == 1])
    real, fake = real[np.isfinite(real)], fake[np.isfinite(fake)]
    if not len(real) or not len(fake):
        return float("nan")
    return float(mannwhitneyu(fake, real).statistic / (len(real) * len(fake)))


def main():
    parser = argparse.ArgumentParser(description="전환 벤치마크에서 배경 정규화 효과를 측정합니다.")
    parser.add_argument("--region", default="face", choices=REGIONS, help="정규화 효과를 자세히 볼 부위")
    args = parser.parse_args()

    config = load_config()
    center = config["benchmark"]["transition_center"]
    signal_root = Path(config["benchmark"]["output_dir"]) / "signals"
    records = [record for record in (clip_stats(path, center) for path in sorted(signal_root.glob("*/*.npz")))
               if record is not None]
    by_kind = defaultdict(list)
    for record in records:
        by_kind[record["kind"]].append(record)
    print(f"클립 {len(records)}개 (real {sum(r['label'] == 0 for r in records)} / fake {sum(r['label'] == 1 for r in records)})")

    # 1) 전환이 각 지표를 몇 배로 키우는가
    print("\n[1] 전환 구간 안 ÷ 바깥 (중앙값). 1 에 가까우면 전환에 흔들리지 않는다는 뜻입니다.")
    print(f"{'전환':<12}{'개수':>5}{'D_bg':>9}" + "".join(f"{'D_' + name:>10}" for name in REGIONS))
    for kind in KIND_ORDER:
        group = by_kind.get(kind, [])
        if not group:
            continue
        print(f"{kind:<12}{len(group):>5}{ratio_median(group, 'bg'):>9.1f}"
              + "".join(f"{ratio_median(group, name):>10.2f}" for name in REGIONS))

    # 2) α 추정 (control 제외)
    transition_records = [record for record in records if record["kind"] != "control"]
    alphas = {name: fit_alpha(transition_records, name) for name in REGIONS}
    print("\n[2] 배경 변화 1 만큼당 부위 변화 α (전환 클립 전체 최소제곱)")
    print("   " + "  ".join(f"{name} {alphas[name]:.2f}" for name in REGIONS))

    # 3) 정규화 전후 증가분
    region = args.region
    alpha = alphas[region]
    print(f"\n[3] {region} 부위의 전환 증가분: 정규화 전 vs R = D_{region} - {alpha:.2f}·D_bg")
    print(f"{'전환':<12}{'정규화 전 Δ':>13}{'정규화 후 Δ':>13}{'제거율':>8}{'남은 배수':>10}")
    for kind in KIND_ORDER:
        group = by_kind.get(kind, [])
        if not group:
            continue
        raw = np.median([record[f"{region}_in"] - record[f"{region}_out"] for record in group])
        residual = np.median([(record[f"{region}_in"] - alpha * record["bg_in"])
                              - (record[f"{region}_out"] - alpha * record["bg_out"]) for record in group])
        removed = 1 - residual / raw if raw else float("nan")
        # 남은 배수: 정규화한 값의 안/밖 비율
        ratios = [(record[f"{region}_in"] - alpha * record["bg_in"]) / (record[f"{region}_out"] - alpha * record["bg_out"])
                  for record in group if (record[f"{region}_out"] - alpha * record["bg_out"]) > 0]
        print(f"{kind:<12}{raw:>+13.4f}{residual:>+13.4f}{removed:>8.0%}{np.median(ratios) if ratios else float('nan'):>10.2f}")

    # 4) real/fake 신호가 남아 있는지 (전환이 없는 control 클립으로 확인)
    controls = by_kind.get("control", [])
    print("\n[4] real vs fake 구분력 (control 클립, AUC). 정규화 뒤에도 비슷해야 신호를 지킨 것입니다.")
    print(f"{'지표':<22}{'정규화 전':>10}{'정규화 후':>10}")
    for name in REGIONS:
        before = auc_of(controls, lambda record, n=name: record[f"{n}_out"])
        after = auc_of(controls, lambda record, n=name: record[f"{n}_out"] - alphas[n] * record["bg_out"])
        print(f"{'D_' + name:<22}{before:>10.3f}{after:>10.3f}")

    # 클립별 값 저장
    out_path = Path(config["paths"]["results"]) / "benchmark" / "clip_stats.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"\n클립별 값 저장: {out_path}")


if __name__ == "__main__":
    main()
