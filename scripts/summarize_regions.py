# ============================================================================
# 부위 신호 요약: real vs fake 비교와 "같은 장면 쌍" 비교
# ----------------------------------------------------------------------------
# preprocessing/region_signals.py 가 만든 summary.csv 를 읽어, 가설 확인용 표를 찍습니다.
#
# 무엇을 보나
#   1) 출처(source)별 중앙값: 조작 기법마다 어느 부위가 얼마나 변하는지
#   2) real vs fake AUC: 지표 하나만으로 둘을 얼마나 가르는지 (0.5 = 전혀 못 가름, 1 = 완벽)
#   3) 같은 장면 쌍 비교: fake 는 특정 real 영상에서 만들어집니다. 같은 장면끼리 빼면
#      배경·인물·조명·촬영 조건이 지워지고 조작 효과만 남습니다.
#        Celeb-DF : Celeb-synthesis/idX_idY_NNNN  ←  Celeb-real/idX_NNNN
#        FF++     : Deepfakes/000_003             ←  original/000
#      real 하나에 fake 가 여러 개 붙으므로 real 마다 "fake 평균 - real" 한 값으로 모읍니다.
#      그래야 영상이 많은 인물·장면 쪽으로 결과가 쏠리지 않습니다.
#
# 지표
#   d_eyes, d_mouth, d_boundary, d_face  : 부위별 프레임 간 변화량 D_region
#   boundary/face, eyes/face, mouth/face : 얼굴 전체 대비 비율. 사진마다 다른 질감·움직임의 영향을 줄입니다.
#   d_bg                                 : 배경 변화량. 같은 장면 쌍에서는 거의 같아야 정상입니다(검산용).
#
# 기본은 학습 분할만 씁니다. 테스트 분할을 보고 설계를 바꾸면 나중 평가가 공정하지 않습니다.
#
# 실행 예 (프로젝트 폴더에서)
#   python scripts/summarize_regions.py --dataset celebdf
#   python scripts/summarize_regions.py --dataset ffpp
#   python scripts/summarize_regions.py --dataset ffpp --only-background-ok
# ============================================================================

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import mannwhitneyu


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

BASE_METRICS = ["d_eyes", "d_mouth", "d_boundary", "d_face", "d_bg"]
# 얼굴 전체 대비 비율 지표: (분자, 분모)
RATIO_METRICS = {
    "boundary/face": ("d_boundary", "d_face"),
    "eyes/face": ("d_eyes", "d_face"),
    "mouth/face": ("d_mouth", "d_face"),
}
METRICS = BASE_METRICS + list(RATIO_METRICS)

# fake 이름에서 원본 영상 번호를 뽑는 규칙
CELEBDF_FAKE = re.compile(r"Celeb-synthesis/(id\d+)_id\d+_(\d+)$")
# 폴더 이름에 숫자가 들어가므로(Face2Face) 폴더 부분은 보지 않고 파일 이름만 봅니다.
FFPP_FAKE = re.compile(r"(\d+)_\d+")


def load_config():
    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return yaml.safe_load(file)


# fake 영상이 어느 real 영상에서 만들어졌는지 돌려줍니다. 짝을 못 찾으면 None 입니다.
def source_video(video_id, dataset):
    if dataset == "celebdf":
        match = CELEBDF_FAKE.match(video_id)
        return f"Celeb-real/{match.group(1)}_{match.group(2)}" if match else None
    match = FFPP_FAKE.fullmatch(video_id.split("/")[-1])
    return f"original/{match.group(1)}" if match else None


# summary.csv 한 줄에서 지표들을 꺼냅니다. 비율 지표는 여기서 계산합니다.
def metrics_of(row):
    values = {}
    for name in BASE_METRICS:
        try:
            values[name] = float(row[name])
        except (KeyError, ValueError):
            values[name] = float("nan")
    for name, (numerator, denominator) in RATIO_METRICS.items():
        bottom = values[denominator]
        values[name] = values[numerator] / bottom if np.isfinite(bottom) and bottom > 0 else float("nan")
    return values


def median_of(rows, name):
    values = np.array([row["metrics"][name] for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else float("nan")


# AUC = fake 하나와 real 하나를 무작위로 골랐을 때 fake 의 지표가 더 클 확률
def auc_of(real_rows, fake_rows, name):
    real = np.array([row["metrics"][name] for row in real_rows], dtype=float)
    fake = np.array([row["metrics"][name] for row in fake_rows], dtype=float)
    real, fake = real[np.isfinite(real)], fake[np.isfinite(fake)]
    if not len(real) or not len(fake):
        return float("nan")
    return float(mannwhitneyu(fake, real).statistic / (len(real) * len(fake)))


# 같은 장면 쌍에서 real 영상마다 (그 영상으로 만든 fake 들의 평균 - real) 을 구합니다.
# 반환값: (차이 목록, 비율 목록)
def paired_differences(pairs, by_id, name):
    differences, ratios = [], []
    for origin, group in pairs.items():
        base = by_id[origin]["metrics"][name]
        values = [row["metrics"][name] for row in group if np.isfinite(row["metrics"][name])]
        if not np.isfinite(base) or not values:
            continue
        differences.append(float(np.mean(values)) - base)
        if base > 0:
            ratios.append(float(np.mean(values)) / base)
    return np.array(differences), np.array(ratios)


def main():
    parser = argparse.ArgumentParser(description="부위 신호 summary.csv 를 요약해 real/fake 차이를 봅니다.")
    parser.add_argument("--dataset", required=True, choices=["celebdf", "ffpp"])
    parser.add_argument("--split", default="train", help="볼 분할 (기본 train)")
    parser.add_argument("--only-background-ok", action="store_true", help="배경 면적이 충분한 영상만 씁니다.")
    args = parser.parse_args()

    config = load_config()
    path = Path(config["paths"]["regions"]) / args.dataset / "summary.csv"
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["error"] or row["split"] != args.split:
                continue
            if args.only_background_ok and row["background_ok"] != "1":
                continue
            row["metrics"] = metrics_of(row)
            rows.append(row)

    reals = [row for row in rows if row["label"] == "0"]
    fakes = [row for row in rows if row["label"] == "1"]
    print(f"[{args.dataset} / {args.split}] real {len(reals)}개, fake {len(fakes)}개  ({path})")

    print("\n[1] 출처별 중앙값")
    print(f"{'source':<17}{'개수':>6}" + "".join(f"{name:>14}" for name in METRICS))
    for source in sorted({row["source"] for row in rows}):
        group = [row for row in rows if row["source"] == source]
        print(f"{source:<17}{len(group):>6}" + "".join(f"{median_of(group, name):>14.4f}" for name in METRICS))

    print("\n[2] real vs fake AUC (0.5 = 못 가름, 1 = 완벽)")
    print(f"{'fake 쪽':<17}{'개수':>6}" + "".join(f"{name:>14}" for name in METRICS))
    fake_sources = sorted({row["source"] for row in fakes})
    groups = [("전체", fakes)] + [(source, [row for row in fakes if row["source"] == source]) for source in fake_sources]
    for label, group in groups:
        if len(groups) == 2 and label == "전체":
            continue  # 기법이 하나뿐이면 전체와 같아서 생략
        print(f"{label:<17}{len(group):>6}" + "".join(f"{auc_of(reals, group, name):>14.3f}" for name in METRICS))

    by_id = {row["video_id"]: row for row in rows}
    pairs = defaultdict(dict)
    for row in fakes:
        origin = source_video(row["video_id"], args.dataset)
        if origin in by_id and by_id[origin]["label"] == "0":
            pairs[row["source"]].setdefault(origin, []).append(row)

    print("\n[3] 같은 장면 쌍: real 영상마다 (fake 평균 - real)")
    print(f"{'기법':<17}{'쌍':>5}{'지표':>15}{'차이 중앙값':>13}{'비율 중앙값':>13}{'fake 가 큼':>11}")
    for method in sorted(pairs):
        for name in METRICS:
            differences, ratios = paired_differences(pairs[method], by_id, name)
            if not len(differences):
                continue
            ratio_text = f"{np.median(ratios):.2f}배" if len(ratios) else "-"
            print(f"{method:<17}{len(differences):>5}{name:>15}{np.median(differences):>+13.4f}"
                  f"{ratio_text:>13}{np.mean(differences > 0):>11.0%}")
    if not pairs:
        print("  (같은 장면 쌍을 찾지 못했습니다)")


if __name__ == "__main__":
    main()
