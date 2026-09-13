# ============================================================================
# 0단계 파일럿 분석: 장면 전환이 시간 기반 탐지기(FTCN)의 오탐을 늘리는가
# ----------------------------------------------------------------------------
# 전체 흐름
#   1) FTCN 채점   : manifest 의 영상(대조군 + 전환 합성본)을 FTCN 으로 채점합니다.
#                    models/ftcn_runner.py 를 별도 프로세스로 실행하고, 결과는 processed/pilot/ftcn/*.npz 에 쌓입니다.
#                    이미 채점한 영상은 건너뛰므로, 중간에 끊겨도 다시 실행하면 이어서 합니다.
#   2) 짝지은 비교 : 합성본마다 같은 프레임 번호의 대조군 점수와 비교해 "전환 때문에 점수가 얼마나 달라졌는지" 계산합니다.
#   3) 요약·검정   : 전환 종류·길이별로 모아 통계 검정을 하고 CSV 로 저장합니다.
#   4) 그래프      : results/pilot/*.png
#
# 입력: processed/pilot/manifest.csv
# 출력: results/pilot/clip_scores.csv          영상마다 FTCN 점수
#       results/pilot/transition_effects.csv   합성본마다 전환 효과
#       results/pilot/summary.csv              전환 종류·길이별 요약과 검정 결과
#       results/pilot/length_trend.csv         전환 길이 추세 검정
#       results/pilot/transition_effect.png, frame_scores_dissolve_*.png
#
# 실행 예 (프로젝트 폴더에서)
#   python scripts/run_pilot.py              채점 + 분석 (영상 90개 기준 약 1시간)
#   python scripts/run_pilot.py --skip-ftcn  이미 있는 채점 결과로 분석만 (몇 초)
#
# 비교 방식 (짝지은 비교)
#   전환 합성본 T 의 프레임은 전환 구간 밖에서 대조군과 원본이 같습니다(synthesize_transitions.py 참고).
#   기준 점수 s_ref(f) 는 f < center 이면 A 대조군, f >= center 이면 B 대조군의 같은 번호 프레임 점수입니다.
#   FTCN 은 32프레임 클립 단위로 보므로 전환 구간을 앞뒤로 32프레임씩 넓힌 "영향 구간"에서 둘을 비교합니다.
#   영향 구간 밖에서는 차이가 0 에 가까워야 합니다(구간 밖 Δ 로 확인).
#
# 효과 척도 (2026-09-13 첫 파일럿 결과를 보고 바꿈)
#   FTCN 점수는 real 영상 대부분이 0.001 근처이고 일부만 0.5 이상입니다. 점수를 그대로 빼면
#   0.001 → 0.01 같은 10배 상승은 거의 0 으로, 점수가 높은 영상의 작은 하락은 크게 잡힙니다.
#   그래서 주 척도는 logit(점수) 차이의 영향 구간 평균(logit Δ)이고, 점수 차이(Δ)는 참고로 함께 저장합니다.
#   이 척도는 결과를 본 뒤 정했으므로, 영상 수를 늘린 본 실험에서는 이 기준을 미리 정한 것으로 씁니다.
#
#   logit(p) = log(p / (1 - p)) : 0~1 사이 확률을 -무한대 ~ +무한대 로 펼친 값입니다.
#     p = 0.001 → -6.9,   p = 0.01 → -4.6,   p = 0.5 → 0,   p = 0.99 → +4.6
#     0.001 → 0.01 (10배) 는 logit 으로 +2.3, 0.5 → 0.6 은 +0.4 입니다.
#
# 판정에 쓰는 값
#   - 전환 종류·길이별: 영상 쌍마다의 logit Δ 가 0 보다 큰지 (Wilcoxon 부호순위 검정, 단측)
#   - 경보율 증가: 영향 구간 프레임이 임계값을 넘는 비율의 (합성본 − 대조군), 같은 검정
#       데모 임계값: FTCN 공식 test_on_raw_video.py 가 결과 영상에 fake 를 표시할 때 쓰는 0.00258
#       real 기준 임계값: 대조군 영상마다 프레임 점수의 95백분위를 구해 그 중앙값
#         (대조군 프레임을 한데 모아 백분위를 구하면 점수가 유난히 높은 영상 하나가 기준을 좌우합니다)
#   - 길이 추세: 같은 영상에서 전환이 길수록 logit Δ 가 커지는지 (Page 추세 검정, 길이가 3개 이상인 종류만)
#
# 통계 용어 짧은 설명
#   p 값 : "실제로는 효과가 없는데 우연히 이만큼 치우친 결과가 나올 확률". 보통 0.05 보다 작으면 유의하다고 봅니다.
#          조건 8개를 한꺼번에 보면 우연히 하나쯤 작아질 수 있으므로, 여러 조건을 볼 때는 더 엄격하게 봐야 합니다.
#   Wilcoxon 부호순위 검정 : 짝지은 차이(영상 쌍마다의 값)들이 0 보다 큰 쪽으로 치우쳤는지 봅니다.
#          차이의 크기 순위와 부호만 써서 정규분포를 가정하지 않습니다. 영상이 적을 때 쓰기 좋습니다.
#          영상 10개가 모두 증가하면 단측 p = 1/1024 ≈ 0.001 이고, 이것이 나올 수 있는 가장 작은 값입니다.
#   단측 검정 : "커졌는가"만 봅니다("달라졌는가"를 보는 양측 검정보다 한 방향 효과를 잘 잡습니다).
#   Page 추세 검정 : 같은 영상에서 조건(디졸브 8, 16, 32, 64)을 순서대로 바꿨을 때 값이 계속 커지는 경향이 있는지 봅니다.
#          영상마다 네 값에 순위를 매겨, 순위가 길이 순서와 얼마나 잘 맞는지로 판단합니다.
#   백분위 : 95백분위 = 값을 작은 순으로 줄 세웠을 때 95% 지점의 값. 그보다 큰 값은 상위 5% 입니다.
#   중앙값 : 한가운데 값. 튀는 값 하나에 덜 흔들려서 평균 대신 씁니다.
# ============================================================================

import argparse
import csv
import math
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

# 그래프를 화면에 띄우지 않고 파일로만 저장하는 모드. pyplot 을 import 하기 "전에" 정해야 합니다.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import page_trend_test, wilcoxon  # noqa: E402


# __file__는 현재 파일 경로입니다. 상위 두 단계인 프로젝트 폴더를 기준으로 설정 파일을 찾습니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
RUNNER_PATH = PROJECT_ROOT / "models" / "ftcn_runner.py"

# FTCN 공식 test_on_raw_video.py 가 SupplyWriter 에 넘기는 fake 판정 기준값입니다.
DEMO_THRESHOLD = 0.002584857167676091
# logit 을 계산할 때 점수가 0 이나 1 에 딱 붙어 무한대가 되지 않도록 자르는 값입니다.
LOGIT_EPS = 1e-6

KIND_LABELS = {"hard_cut": "하드컷", "fade": "페이드", "dissolve": "디졸브"}

# 그래프 색. 전환 종류는 순서 없는 범주라 기본 팔레트 앞 세 칸을 고정 순서로 씁니다.
# 글자는 데이터 색이 아니라 회색 계열(INK, INK_SECONDARY)로 써서 읽기 쉽게 합니다.
SURFACE = "#fcfcfb"  # 배경
INK = "#0b0b0b"  # 제목
INK_SECONDARY = "#52514e"  # 설명 글자
MUTED = "#898781"  # 보조 선(대조군)
GRID = "#e1e0d9"  # 격자
AXIS = "#c3c2b7"  # 축
WASH = "#f0efec"  # 전환 구간 음영
KIND_COLORS = {"hard_cut": "#2a78d6", "fade": "#eb6834", "dissolve": "#1baf7a"}


# 여러 스크립트가 같은 경로와 기준값을 쓰도록 YAML 설정을 딕셔너리로 읽습니다.
def load_config():
    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return yaml.safe_load(file)


# FTCN 실행기를 별도 프로세스로 실행합니다. 이미 결과가 있는 영상은 실행기가 건너뜁니다.
# sys.executable 은 지금 이 스크립트를 실행 중인 파이썬(.venv 의 python.exe) 경로라, 같은 환경으로 실행됩니다.
def run_ftcn(manifest_path, ftcn_dir, clip_batch):
    command = [
        sys.executable, str(RUNNER_PATH),
        "--manifest", str(manifest_path),
        "--out-dir", str(ftcn_dir),
        "--clip-batch", str(clip_batch),
    ]
    # 지금 환경 변수를 복사하고, 한글 출력이 깨지지 않도록 PYTHONIOENCODING 만 추가합니다.
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    completed = subprocess.run(command, env=environment)
    if completed.returncode != 0:
        raise SystemExit("FTCN 실행이 실패했습니다. 위 로그를 확인하세요.")


# 영상 하나의 FTCN 결과(.npz)를 딕셔너리로 읽습니다. 없으면 None 입니다.
# with np.load(...) 로 열고 모든 배열을 꺼내 두면, 파일을 닫은 뒤에도 값을 쓸 수 있습니다.
def load_result(ftcn_dir, clip_id):
    path = ftcn_dir / f"{clip_id}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


# 점수(0~1)를 logit 으로 바꿉니다. nan 은 nan 으로 남습니다.
# 배열 전체에 한 번에 계산됩니다(numpy 의 벡터 연산).
def logit(scores):
    scores = np.clip(np.asarray(scores, dtype=np.float64), LOGIT_EPS, 1 - LOGIT_EPS)
    return np.log(scores / (1 - scores))


# 전환이 점수에 영향을 줄 수 있는 프레임 구간 [시작, 끝) 입니다.
# 32프레임 클립이 전환 프레임을 하나라도 포함하면 그 클립이 덮는 프레임 전체의 점수가 달라질 수 있습니다.
#   예) 디졸브 32, 전환 구간 [112, 144), 여백 32 → 영향 구간 [80, 176)
#   하드컷은 구간 길이가 0 이라 중심에서 앞뒤로 32프레임: [96, 160)
def effect_region(row, margin, frame_count):
    center = int(row["center"])
    window_start = int(row["window_start"])
    window_end = int(row["window_end"])

    if window_start == window_end:
        start, end = center - margin, center + margin
    else:
        start, end = window_start - margin, window_end + margin

    # 영상 범위(0 ~ frame_count)를 벗어나지 않게 자릅니다.
    return max(0, start), min(frame_count, end)


# FTCN 이 전환을 실제로 입력으로 봤는지 확인합니다.
# 얼굴 추적이 전환에서 끊기면 클립이 전환을 넘지 않으므로, FTCN 은 전환을 보지 못한 채 점수를 냅니다.
#   하드컷: 전환 앞(center 미만)과 뒤(center 이상) 프레임을 함께 가진 클립이 하나라도 있어야 "봤다"
#   페이드/디졸브: 섞인 프레임(전환 구간 안)을 하나라도 가진 클립이 있으면 "봤다"
def saw_transition(result, row):
    clip_frames = result["clip_frames"]
    if len(clip_frames) == 0:
        return False

    # axis=1: 클립마다(행마다) 가장 작은/큰 프레임 번호
    lowest = clip_frames.min(axis=1)
    highest = clip_frames.max(axis=1)
    center = int(row["center"])
    window_start = int(row["window_start"])
    window_end = int(row["window_end"])

    if row["kind"] == "hard_cut":
        # & 는 배열끼리 원소별 "그리고". np.any: 하나라도 True 면 True
        return bool(np.any((lowest < center) & (highest >= center)))
    return bool(np.any((lowest < window_end) & (highest >= window_start)))


# 유한한 값만 모아 평균을 냅니다. 없으면 nan 입니다.
# np.isfinite: nan 이나 무한대가 아닌 값만 True. 채점되지 않은 프레임(nan)을 빼기 위해 씁니다.
def finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else math.nan


# 유한한 값만 모아 중앙값을 냅니다. 없으면 nan 입니다.
def finite_median(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else math.nan


# 0 보다 큰 값의 비율입니다(유한한 값 기준). 예) 10개 중 7개가 양수면 0.7
def positive_share(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float((values > 0).mean()) if len(values) else math.nan


# 값들이 0 보다 크다는 단측 Wilcoxon 부호순위 검정의 p 값입니다. 계산할 수 없으면 nan 입니다.
# 값이 2개 미만이거나 모두 0 이면 검정할 수 없습니다.
def one_sided_p(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2 or not np.any(values != 0):
        return math.nan
    try:
        return float(wilcoxon(values, alternative="greater").pvalue)
    except ValueError:
        return math.nan


# 임계값을 넘는 비율을 유한한 값만 대상으로 계산합니다.
# (values > threshold) 는 True/False 배열이고, 평균을 내면 True 의 비율이 됩니다.
def exceed_rate(values, threshold):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float((values > threshold).mean()) if len(values) else math.nan


# 대조군으로 경보율 임계값을 정합니다. 반환값은 {"demo": 값, "real95": 값} 입니다.
def alarm_thresholds(control_results):
    per_video = []
    for result in control_results:
        scores = result["frame_scores"]
        scores = scores[np.isfinite(scores)]
        if len(scores):
            # 이 영상의 프레임 점수 중 상위 5% 가 시작되는 값
            per_video.append(float(np.percentile(scores, 95)))
    if not per_video:
        raise SystemExit("대조군 점수가 없어 임계값을 정할 수 없습니다.")
    # 영상마다 구한 값의 중앙값을 써서, 점수가 유난히 높은 영상 하나에 기준이 끌려가지 않게 합니다.
    return {"demo": DEMO_THRESHOLD, "real95": float(np.median(per_video))}


# 전환 합성본 하나의 효과를 계산합니다. 반환값은 transition_effects.csv 의 한 줄이 됩니다.
def transition_effect(row, results, margin, thresholds):
    treatment = results[row["clip_id"]]
    base = results[f"{row['base']}__control"]
    partner = results[f"{row['partner']}__control"]

    # 세 영상 중 가장 짧은 길이에 맞춥니다(보통 모두 256).
    frame_count = min(len(treatment["frame_scores"]), len(base["frame_scores"]), len(partner["frame_scores"]))
    center = int(row["center"])

    # 기준 점수: center 앞은 A 대조군, 뒤는 B 대조군의 같은 번호 프레임 점수를 이어 붙입니다.
    treatment_scores = treatment["frame_scores"][:frame_count].astype(np.float64)
    reference = np.concatenate([base["frame_scores"][:center], partner["frame_scores"][center:frame_count]])
    reference = reference.astype(np.float64)

    start, end = effect_region(row, margin, frame_count)
    # 프레임마다의 차이. 양수면 합성본이 더 fake 쪽으로 판단됐다는 뜻입니다.
    difference = treatment_scores - reference
    logit_difference = logit(treatment_scores) - logit(reference)

    window = slice(int(row["window_start"]), int(row["window_end"]))

    effect = {
        "clip_id": row["clip_id"],
        "base": row["base"],
        "partner": row["partner"],
        "kind": row["kind"],
        "length": int(row["length"]),
        # CSV 에서 읽은 값은 문자열이라 "True" 와 비교해 True/False 로 바꿉니다.
        "partner_resized": row["partner_resized"] == "True",
        "fps_mismatch": row["fps_mismatch"] == "True",
        "saw_transition": saw_transition(treatment, row),
        "full_track": bool(treatment["full_track"]),
        "segments": int(len(treatment["segments"])),
        # 전환 구간 안에서 얼굴을 못 찾은 프레임 수 (페이드의 어두운 프레임에서 생김)
        "no_face_in_window": int((treatment["faces"][window] == 0).sum()),
        # 영향 구간에서 점수가 있는 프레임의 비율 (1.0 이면 모든 프레임이 채점됨)
        "coverage_treatment": float(np.isfinite(treatment_scores[start:end]).mean()),
        "coverage_reference": float(np.isfinite(reference[start:end]).mean()),
        # 주 척도: 영향 구간의 logit 차이 평균
        "logit_delta_window": finite_mean(logit_difference[start:end]),
        # 확인용: 영향 구간 밖의 logit 차이 평균 (0 에 가까워야 정상)
        "logit_delta_outside": finite_mean(np.concatenate([logit_difference[:start], logit_difference[end:]])),
        # 참고: 점수를 그대로 뺀 차이
        "delta_window": finite_mean(difference[start:end]),
        "delta_outside": finite_mean(np.concatenate([difference[:start], difference[end:]])),
        # 참고: 영상 점수(모든 클립 평균)의 차이. 대조군은 A, B 의 평균과 비교합니다.
        "delta_video": float(treatment["video_score"]) - (float(base["video_score"]) + float(partner["video_score"])) / 2,
    }
    # 임계값마다: 합성본과 대조군에서 영향 구간 프레임이 fake 로 판정되는 비율, 그리고 그 증가량
    for name, threshold in thresholds.items():
        treated = exceed_rate(treatment_scores[start:end], threshold)
        referenced = exceed_rate(reference[start:end], threshold)
        effect[f"alarm_treatment_{name}"] = treated
        effect[f"alarm_reference_{name}"] = referenced
        effect[f"alarm_increase_{name}"] = treated - referenced
    return effect


# 전환 종류·길이마다 효과를 요약합니다. 반환값은 summary.csv 의 줄 목록입니다.
def summarize(effects, groups, threshold_names):
    summary = []
    for kind, length in groups:
        items = [e for e in effects if e["kind"] == kind and e["length"] == length]
        logit_deltas = [e["logit_delta_window"] for e in items]

        item = {
            "kind": kind,
            "length": length,
            "videos": len(items),
            "saw_transition": sum(e["saw_transition"] for e in items),
            "full_track": sum(e["full_track"] for e in items),
            "median_logit_delta": finite_median(logit_deltas),
            "share_logit_delta_positive": positive_share(logit_deltas),
            "p_logit_delta": one_sided_p(logit_deltas),
            "median_logit_delta_outside": finite_median([e["logit_delta_outside"] for e in items]),
            "median_delta_window": finite_median([e["delta_window"] for e in items]),
            "median_delta_video": finite_median([e["delta_video"] for e in items]),
        }
        for name in threshold_names:
            increases = [e[f"alarm_increase_{name}"] for e in items]
            item[f"alarm_treatment_{name}"] = finite_mean([e[f"alarm_treatment_{name}"] for e in items])
            item[f"alarm_reference_{name}"] = finite_mean([e[f"alarm_reference_{name}"] for e in items])
            item[f"median_alarm_increase_{name}"] = finite_median(increases)
            item[f"share_alarm_increase_{name}"] = positive_share(increases)
            item[f"p_alarm_increase_{name}"] = one_sided_p(increases)
        summary.append(item)
    return summary


# 같은 영상에서 전환이 길수록 logit Δ 가 커지는지 Page 추세 검정으로 봅니다. 길이가 3개 이상인 종류만 합니다.
# 검정에 넣는 표(matrix): 행 = 영상, 열 = 길이(짧은 것부터). 예) 영상 10개 × 디졸브 4길이 = [10, 4]
def length_trends(effects, groups):
    trends = []
    for kind in KIND_LABELS:
        lengths = [length for group_kind, length in groups if group_kind == kind]
        if len(lengths) < 3:
            continue

        # {기준 영상: {길이: logit Δ}} 로 모읍니다.
        by_video = {}
        for e in effects:
            if e["kind"] == kind and np.isfinite(e["logit_delta_window"]):
                by_video.setdefault(e["base"], {})[e["length"]] = e["logit_delta_window"]
        # 모든 길이의 값이 다 있는 영상만 씁니다(빈칸이 있으면 순위를 매길 수 없음).
        complete = [values for values in by_video.values() if all(length in values for length in lengths)]
        if len(complete) < 2:
            continue

        matrix = np.array([[values[length] for length in lengths] for values in complete])
        result = page_trend_test(matrix)
        trends.append({
            "kind": kind,
            "lengths": " ".join(str(length) for length in lengths),
            "videos": len(complete),
            "page_L": float(result.statistic),
            "p_increasing": float(result.pvalue),
            "median_logit_delta_by_length": " ".join(
                f"{length}:{np.median(matrix[:, index]):.3f}" for index, length in enumerate(lengths)
            ),
        })
    return trends


# 딕셔너리 목록을 CSV 로 저장합니다. 열 이름은 첫 줄의 키를 씁니다.
# encoding="utf-8-sig": 엑셀에서 열었을 때 한글이 깨지지 않도록 파일 맨 앞에 표시(BOM)를 넣습니다.
def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# 그래프와 표에 쓸 이름. 예) ("dissolve", 32) → "디졸브 32", ("hard_cut", 0) → "하드컷"
def group_label(kind, length):
    return KIND_LABELS[kind] if kind == "hard_cut" else f"{KIND_LABELS[kind]} {length}"


# 한글이 네모로 깨지지 않도록 윈도우 기본 한글 글꼴을 먼저 쓰게 합니다.
def setup_fonts():
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    # 마이너스 기호를 유니코드 대신 일반 하이픈으로 그려서, 한글 글꼴에서 깨지지 않게 합니다.
    plt.rcParams["axes.unicode_minus"] = False


# 축과 격자를 차분하게 맞춥니다: 가는 실선 격자, 위·오른쪽 테두리 없음, 글자는 회색 계열.
def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=AXIS, labelcolor=INK_SECONDARY, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6, linestyle="-")
    # 격자를 점과 선 뒤로 보냅니다.
    ax.set_axisbelow(True)


# 그림 1: 전환 종류·길이별 영상마다의 logit Δ. 채운 점은 FTCN 이 전환을 본 경우, 빈 점은 추적이 끊겨 못 본 경우입니다.
def plot_effects(effects, groups, path):
    # figsize 는 인치 단위, dpi 는 인치당 픽셀 수. 9.5 x 150 = 가로 1425 픽셀
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    # 0 기준선: 이 선보다 위면 전환 때문에 더 fake 쪽으로 판단됐다는 뜻
    ax.axhline(0, color=AXIS, linewidth=1.0, zorder=1)

    for x, (kind, length) in enumerate(groups):
        items = [
            e for e in effects
            if e["kind"] == kind and e["length"] == length and np.isfinite(e["logit_delta_window"])
        ]
        if not items:
            continue
        values = np.array([e["logit_delta_window"] for e in items])
        saw = np.array([e["saw_transition"] for e in items])
        # 점들이 한 줄로 겹치지 않게 x 위치를 -0.2 ~ +0.2 로 조금씩 옆으로 벌립니다.
        offsets = np.linspace(-0.2, 0.2, len(values)) if len(values) > 1 else np.zeros(1)
        color = KIND_COLORS[kind]

        # saw 가 True 인 점은 색을 채우고, False(~saw) 인 점은 속이 빈 원으로 그립니다.
        #   zorder: 겹칠 때 위에 그려질 순서(클수록 위)
        ax.scatter(x + offsets[saw], values[saw], s=36, color=color, edgecolors=SURFACE, linewidths=1.2, zorder=3)
        ax.scatter(x + offsets[~saw], values[~saw], s=36, facecolors=SURFACE, edgecolors=color, linewidths=1.4, zorder=3)
        # 중앙값을 짧은 가로선으로 표시합니다.
        median = float(np.median(values))
        ax.plot([x - 0.32, x + 0.32], [median, median], color=INK, linewidth=1.6, solid_capstyle="round", zorder=4)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([group_label(kind, length) for kind, length in groups])
    ax.set_ylabel("logit Δ (영향 구간 평균)", color=INK_SECONDARY, fontsize=9)

    # 범례: 색(전환 종류), 빈 원(전환을 못 봄), 굵은 선(중앙값)의 뜻을 적습니다.
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor=KIND_COLORS[kind],
               markeredgecolor=SURFACE, label=KIND_LABELS[kind])
        for kind in KIND_COLORS
    ]
    handles.append(Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor=SURFACE,
                          markeredgecolor=MUTED, label="FTCN 이 전환을 못 봄 (추적 끊김)"))
    handles.append(Line2D([], [], color=INK, linewidth=1.6, label="중앙값"))
    legend = ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8, ncol=5,
                       bbox_to_anchor=(0, 1.13))
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    # fig.text 의 좌표는 그림 전체 기준(0~1). (0.01, 0.985) = 왼쪽 위
    fig.text(0.01, 0.985, "장면 전환을 넣었을 때 FTCN 점수 변화", fontsize=12, color=INK, va="top", weight="bold")
    fig.text(0.01, 0.935, "점 하나 = real 영상 한 쌍. 0 보다 위면 전환 때문에 FTCN 이 더 fake 쪽으로 판단했다는 뜻 (logit 척도)",
             fontsize=8.5, color=INK_SECONDARY, va="top")
    # 제목 자리를 남기고(위쪽 13%) 나머지에 그래프를 배치합니다.
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# 그림 2: 한 전환 조건에서 영상마다 프레임 점수. 합성본(색)과 같은 번호의 대조군 점수(회색)를 겹쳐 봅니다.
# 영상 하나당 작은 그래프 하나(small multiples)로, 5칸씩 줄을 바꿔 배치합니다.
def plot_frame_scores(rows, results, kind, length, path):
    treatments = [
        row for row in rows
        if row["role"] == "treatment" and row["kind"] == kind and int(row["length"]) == length
        and row["clip_id"] in results
        and f"{row['base']}__control" in results and f"{row['partner']}__control" in results
    ]
    if not treatments:
        return

    columns = 5
    # 올림 나눗셈: 영상 12개면 3줄
    lines = math.ceil(len(treatments) / columns)
    # sharex/sharey: 모든 작은 그래프가 같은 축 범위를 써서 서로 비교하기 쉽게 합니다.
    # squeeze=False: 줄이 하나여도 axes 를 항상 2차원 배열로 돌려받습니다.
    fig, axes = plt.subplots(lines, columns, figsize=(columns * 2.5, lines * 2.1 + 0.8), dpi=150,
                             sharex=True, sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    color = KIND_COLORS[kind]

    # axes.flat: 2차원 배열을 한 줄로 펴서 차례로 꺼냅니다.
    for ax, row in zip(axes.flat, treatments):
        treatment = results[row["clip_id"]]["frame_scores"]
        base = results[f"{row['base']}__control"]["frame_scores"]
        partner = results[f"{row['partner']}__control"]["frame_scores"]
        center = int(row["center"])
        count = min(len(treatment), len(base), len(partner))
        reference = np.concatenate([base[:center], partner[center:count]])

        style_axes(ax)
        window_start, window_end = int(row["window_start"]), int(row["window_end"])
        if window_end > window_start:
            # 전환 구간을 옅은 회색 띠로 칠합니다.
            ax.axvspan(window_start, window_end, color=WASH, zorder=0)
        else:
            # 하드컷은 구간 대신 세로선 하나로 표시합니다.
            ax.axvline(center, color=AXIS, linewidth=0.8, zorder=0)
        ax.plot(np.arange(count), reference, color=MUTED, linewidth=1.0, zorder=2)
        ax.plot(np.arange(count), treatment[:count], color=color, linewidth=1.2, zorder=3)
        ax.set_ylim(0, 1)
        ax.set_title(f"{row['base']} → {row['partner']}", fontsize=8, color=INK_SECONDARY)

    # 영상 수가 5의 배수가 아니면 남는 빈 칸을 숨깁니다.
    for ax in list(axes.flat)[len(treatments):]:
        ax.set_visible(False)

    handles = [
        Line2D([], [], color=color, linewidth=1.2, label="전환 합성본"),
        Line2D([], [], color=MUTED, linewidth=1.0, label="대조군 (전환 없음)"),
        Patch(facecolor=WASH, edgecolor="none", label="전환 구간"),
    ]
    legend = fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=8, ncol=3)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.text(0.01, 0.99, f"프레임별 FTCN 점수 — {group_label(kind, length)}", fontsize=11, color=INK,
             va="top", weight="bold")
    fig.supxlabel("프레임 번호", fontsize=8, color=INK_SECONDARY)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# 표에서 p 값을 보기 좋게 적습니다. 계산할 수 없는 값(nan)은 "-" 로 표시합니다.
def format_p(value):
    return "  -  " if not np.isfinite(value) else f"{value:.3f}"


def main():
    parser = argparse.ArgumentParser(description="0단계 파일럿을 실행하고 결과를 정리합니다.")
    parser.add_argument("--clip-batch", type=int, default=4, help="FTCN 이 한 번에 추론할 클립 수")
    parser.add_argument("--skip-ftcn", action="store_true", help="FTCN 을 돌리지 않고 이미 있는 결과만 분석합니다.")
    args = parser.parse_args()

    config = load_config()
    pilot_dir = Path(config["paths"]["processed"]) / "pilot"
    manifest_path = pilot_dir / "manifest.csv"
    ftcn_dir = pilot_dir / "ftcn"
    results_dir = Path(config["paths"]["results"]) / "pilot"

    with open(manifest_path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    # 1) FTCN 채점
    if not args.skip_ftcn:
        run_ftcn(manifest_path, ftcn_dir, args.clip_batch)

    # 2) 채점 결과 읽기. 실패했거나 없는 영상은 분석에서 빠집니다.
    results = {}
    failed = []
    for row in rows:
        result = load_result(ftcn_dir, row["clip_id"])
        if result is None or str(result["error"]):
            failed.append(row["clip_id"])
            continue
        results[row["clip_id"]] = result
    if failed:
        print(f"결과가 없거나 실패한 영상 {len(failed)}개: {failed[:5]}")

    # 3) 영상마다 FTCN 점수 표 (clip_scores.csv)
    clip_rows = []
    for row in rows:
        result = results.get(row["clip_id"])
        if result is None:
            continue
        clip_rows.append({
            "clip_id": row["clip_id"],
            "role": row["role"],
            "base": row["base"],
            "partner": row["partner"],
            "kind": row["kind"],
            "length": int(row["length"]),
            "video_score": float(result["video_score"]),
            "frames": int(result["n_frames"]),
            "scored_frames": int((result["frame_hits"] > 0).sum()),
            "no_face_frames": int((result["faces"] == 0).sum()),
            "full_track": bool(result["full_track"]),
            "segments": int(len(result["segments"])),
        })
    write_csv(results_dir / "clip_scores.csv", clip_rows)

    # 4) 대조군으로 경보율 임계값 정하기
    controls = [row for row in rows if row["role"] == "control" and row["clip_id"] in results]
    thresholds = alarm_thresholds([results[row["clip_id"]] for row in controls])
    control_videos = [float(results[row["clip_id"]]["video_score"]) for row in controls]

    # 영향 구간 여백 = FTCN 클립 길이(32). 결과 파일의 clip_frames 모양 [클립 수, 32] 에서 가져옵니다.
    # next(iter(딕셔너리.values())) 는 딕셔너리의 첫 번째 값을 꺼내는 관용 표현입니다.
    margin = next(iter(results.values()))["clip_frames"].shape[1]

    # 5) 합성본마다 전환 효과 (transition_effects.csv). 짝이 되는 대조군 결과가 모두 있어야 계산합니다.
    effects = []
    for row in rows:
        if row["role"] != "treatment":
            continue
        needed = (row["clip_id"], f"{row['base']}__control", f"{row['partner']}__control")
        if all(name in results for name in needed):
            effects.append(transition_effect(row, results, margin, thresholds))
    write_csv(results_dir / "transition_effects.csv", effects)

    # 6) 전환 종류·길이별 요약과 길이 추세 (summary.csv, length_trend.csv)
    groups = []
    for kind, lengths in config["pilot"]["transitions"].items():
        for length in lengths:
            groups.append((kind, 0 if kind == "hard_cut" else int(length)))
    summary = summarize(effects, groups, list(thresholds))
    write_csv(results_dir / "summary.csv", summary)
    trends = length_trends(effects, groups)
    write_csv(results_dir / "length_trend.csv", trends)

    # 7) 그래프
    setup_fonts()
    plot_effects(effects, groups, results_dir / "transition_effect.png")
    for kind, length in groups:
        if kind == "dissolve":
            plot_frame_scores(rows, results, kind, length, results_dir / f"frame_scores_dissolve_{length:02d}.png")

    # 8) 화면에 요약 출력
    print()
    print(f"대조군 {len(controls)}개 | 영상 점수 {np.min(control_videos):.3f} ~ {np.max(control_videos):.3f} "
          f"(중앙값 {np.median(control_videos):.3f}) | 영향 구간 여백 {margin}프레임")
    print(f"경보율 임계값: 데모 {thresholds['demo']:.5f}, real 기준(영상별 95백분위의 중앙값) {thresholds['real95']:.5f}")
    print()
    print(f"{'전환':<10}{'영상':>4}{'추적 유지':>9}{'logit Δ':>9}{'증가':>6}{'p':>7}"
          f"{'데모 경보 증가':>14}{'p':>7}{'real95 경보 증가':>16}{'p':>7}{'구간 밖 logit Δ':>16}")
    for item in summary:
        print(
            f"{group_label(item['kind'], item['length']):<10}{item['videos']:>4}"
            f"{item['full_track']:>6}/{item['videos']:<2}"
            f"{item['median_logit_delta']:>9.2f}{item['share_logit_delta_positive']:>6.1f}{format_p(item['p_logit_delta']):>7}"
            f"{item['median_alarm_increase_demo']:>14.2f}{format_p(item['p_alarm_increase_demo']):>7}"
            f"{item['median_alarm_increase_real95']:>16.2f}{format_p(item['p_alarm_increase_real95']):>7}"
            f"{item['median_logit_delta_outside']:>16.2f}"
        )
    for trend in trends:
        print(f"\n{KIND_LABELS[trend['kind']]} 길이 추세 (영상 {trend['videos']}개): Page L={trend['page_L']:.1f}, "
              f"p={trend['p_increasing']:.4g} | 길이별 logit Δ 중앙값 {trend['median_logit_delta_by_length']}")
    print()
    print(f"결과 저장: {results_dir}")


# 이 파일을 직접 실행할 때만 main() 을 부릅니다.
if __name__ == "__main__":
    main()
