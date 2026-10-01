# ============================================================================
# 3단계 벤치마크: 장면 전환이 FTCN 점수를 얼마나 흔드는가
# ----------------------------------------------------------------------------
# 묻는 것
#   같은 기준 영상에서 만든 "대조군(전환 없음)"과 "전환 합성본"의 FTCN 점수를 짝지어 비교합니다.
#     real 클립의 점수가 오르면  → 오탐(장면 전환을 조작으로 착각)
#     fake 클립의 점수가 내리면  → 미탐(장면 전환이 조작을 가림)
#
# 비교 방법 (0단계 파일럿과 같은 방식, scripts/run_pilot.py 의 함수를 그대로 씁니다)
#   - 점수는 0 근처에 몰려 있어 그대로 빼면 차이가 묻힙니다. logit 으로 바꿔서 뺍니다.
#   - FTCN 은 32프레임 클립 단위로 보므로, 전환 구간을 앞뒤로 32프레임 넓힌 "영향 구간"의 프레임 점수만 평균냅니다.
#   - 얼굴 추적이 전환에서 끊기면 FTCN 은 전환을 입력으로 보지 못합니다(하드컷에서 흔합니다).
#     그런 클립은 "전환을 본 클립"과 나눠서 봅니다.
#
# 입력: benchmark.output_dir/manifest.csv 와 ftcn/<source>/<clip_id>.npz (models/ftcn_runner.py 결과)
# 출력: 화면 표 + paths.results/benchmark/ftcn_effect.csv (클립마다 대조군 대비 점수 변화)
#
# 실행 예 (프로젝트 폴더에서)
#   python scripts/analyze_benchmark_ftcn.py
# ============================================================================

import argparse
import csv
import importlib.util
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# FTCN 공식 데모가 fake 로 표시할 때 쓰는 기준값입니다(0단계 파일럿과 같은 값).
DEMO_THRESHOLD = 0.002584857167676091
# FTCN 이 한 번에 보는 클립 길이. 전환 앞뒤로 이만큼을 "영향 구간"으로 잡습니다.
CLIP_MARGIN = 32

# scripts 폴더는 패키지가 아니라서 파일 경로로 직접 불러옵니다.
_spec = importlib.util.spec_from_file_location("run_pilot", PROJECT_ROOT / "scripts" / "run_pilot.py")
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)


def load_config():
    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return yaml.safe_load(file)


# FTCN 결과 .npz 를 읽습니다. 없거나 오류가 적혀 있으면 None 입니다.
# clip_id 는 기법마다 겹치므로(Deepfakes/000_003 과 Face2Face/000_003) source 하위 폴더에서 찾습니다.
def load_result(ftcn_dir, row):
    path = Path(ftcn_dir) / row["source"] / f"{row['clip_id']}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        if str(data["error"]).strip():
            return None
        return {
            "frame_scores": data["frame_scores"],
            "clip_frames": data["clip_frames"],
            "video_score": float(data["video_score"]),
            "full_track": bool(data["full_track"]),
        }


# 영향 구간 안의 프레임 점수를 logit 으로 바꿔 평균냅니다.
def region_logit(result, start, end):
    scores = np.asarray(result["frame_scores"], dtype=np.float64)[start:end]
    scores = scores[np.isfinite(scores)]
    return float(np.mean(pilot.logit(scores))) if len(scores) else float("nan")


# 짝지은 차이에 부호순위 검정을 합니다(0 보다 큰 쪽으로). 표본이 적으면 nan 입니다.
def wilcoxon_p(differences):
    values = np.asarray([value for value in differences if np.isfinite(value) and value != 0])
    if len(values) < 6:
        return float("nan")
    return float(wilcoxon(values, alternative="greater").pvalue)


def main():
    parser = argparse.ArgumentParser(description="전환이 FTCN 점수를 얼마나 흔드는지 대조군과 짝지어 비교합니다.")
    parser.add_argument("--margin", type=int, default=CLIP_MARGIN, help="전환 앞뒤로 영향 구간에 포함할 프레임 수")
    args = parser.parse_args()

    config = load_config()
    benchmark = config["benchmark"]
    root = Path(benchmark["output_dir"])
    ftcn_dir = root / "ftcn"
    center = benchmark["transition_center"]

    with open(root / "manifest.csv", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    # 기준 영상마다 대조군과 전환 클립을 모읍니다.
    controls, transitions = {}, defaultdict(list)
    for row in rows:
        key = (row["source"], row["base_id"])
        if row["kind"] == "control":
            controls[key] = row
        else:
            transitions[key].append(row)

    records = []
    for key, group in transitions.items():
        control_row = controls.get(key)
        if control_row is None:
            continue
        control = load_result(ftcn_dir, control_row)
        if control is None:
            continue
        for row in group:
            result = load_result(ftcn_dir, row)
            if result is None:
                continue
            frame_count = int(row["frames"])
            # run_pilot 의 함수가 쓰는 열만 추린 딕셔너리입니다(전환 종류·구간·중심).
            window = {"kind": row["kind"], "center": center,
                      "window_start": row["window_start"], "window_end": row["window_end"]}
            start, end = pilot.effect_region(window, args.margin, frame_count)
            records.append({
                "clip_id": row["clip_id"],
                "source": row["source"],
                "label": int(row["label"]),
                "kind": row["kind"] + (row["length"] if row["length"] != "0" else ""),
                "saw_transition": int(pilot.saw_transition(result, window)),
                "control_logit": region_logit(control, start, end),
                "clip_logit": region_logit(result, start, end),
                "control_video_score": control["video_score"],
                "video_score": result["video_score"],
            })
            records[-1]["delta_logit"] = records[-1]["clip_logit"] - records[-1]["control_logit"]

    if not records:
        print("비교할 결과가 없습니다. FTCN 채점이 끝났는지 확인하세요.")
        return
    print(f"짝지어 비교한 클립 {len(records)}개 (기준 영상 {len(controls)}개 중 결과가 있는 것만)")

    # 전환 종류별 · 정답별 표
    print(f"\n{'전환':<12}{'정답':<6}{'쌍':>5}{'전환 봄':>8}{'logit 변화 중앙값':>18}{'오른 비율':>10}{'p값':>9}"
          f"{'경보율 대조군→전환':>20}")
    kinds = ["hard_cut", "fade8", "fade16", "fade32", "dissolve8", "dissolve16", "dissolve32", "dissolve64"]
    for kind in kinds:
        for label, name in ((0, "real"), (1, "fake")):
            group = [record for record in records if record["kind"] == kind and record["label"] == label]
            if not group:
                continue
            deltas = np.array([record["delta_logit"] for record in group])
            deltas = deltas[np.isfinite(deltas)]
            if not len(deltas):
                continue
            control_alarm = np.mean([record["control_video_score"] >= DEMO_THRESHOLD for record in group])
            clip_alarm = np.mean([record["video_score"] >= DEMO_THRESHOLD for record in group])
            print(f"{kind:<12}{name:<6}{len(group):>5}{np.mean([r['saw_transition'] for r in group]):>8.0%}"
                  f"{np.median(deltas):>+18.3f}{np.mean(deltas > 0):>10.0%}{wilcoxon_p(deltas):>9.4f}"
                  f"{control_alarm:>12.0%} → {clip_alarm:.0%}")

    out_path = Path(config["paths"]["results"]) / "benchmark" / "ftcn_effect.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"\n클립별 값 저장: {out_path}")


if __name__ == "__main__":
    main()
