# ============================================================================
# 4단계: 샷 경계 검출기가 합성 전환을 얼마나 잡아내는가
# ----------------------------------------------------------------------------
# 왜 재는가
#   "장면 전환이 문제라면 샷 경계를 찾아 잘라 쓰면 되지 않나" 라는 반론에 답하기 위해서입니다.
#   벤치마크는 전환을 직접 넣었기 때문에 위치를 정확히 알고 있어, 검출률과 위치 오차를 정확히 잴 수 있습니다.
#
# 쓰는 검출기 (PySceneDetect, 실무에서 가장 널리 쓰입니다)
#   ContentDetector  : 이웃 프레임의 색·밝기 차이가 임계값을 넘으면 경계로 봅니다. 하드컷을 잘 잡습니다.
#                      임계값을 여러 개로 돌려 "민감하게 하면 디졸브도 잡히는지, 대신 오검출은 얼마나 느는지" 봅니다.
#   ThresholdDetector: 화면이 검게 어두워지는 구간을 찾습니다. 검은 화면을 지나는 페이드를 잡습니다.
#
# 판정 기준
#   전환 구간 [시작, 끝) 의 앞뒤 tolerance(8) 프레임 안에 경계가 하나라도 있으면 "검출"로 봅니다.
#   하드컷은 길이가 0 이라 전환 지점 앞뒤 tolerance 프레임 안이면 검출입니다.
#   전환이 없는 control 클립에서 경계가 잡히면 "오검출"입니다.
#
# 입력: benchmark.output_dir/manifest.csv 와 클립 mp4
# 출력: paths.results/benchmark/shot_detection.csv (클립마다 검출된 경계)와 화면 요약표
#
# 실행 예 (프로젝트 폴더에서)
#   python scripts/detect_shots_benchmark.py --limit 20   시험
#   python scripts/detect_shots_benchmark.py              전체
# ============================================================================

import argparse
import csv
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 내용 기반 검출기의 임계값. 27 은 PySceneDetect 기본값이고, 낮출수록 민감해집니다.
CONTENT_THRESHOLDS = [27.0, 15.0, 8.0]
# 전환 구간에서 이만큼 떨어진 경계까지 "맞게 찾았다"로 봅니다.
TOLERANCE = 8
TRANSITION_KINDS = ["hard_cut", "fade8", "fade16", "fade32", "dissolve8", "dissolve16", "dissolve32", "dissolve64"]


def load_config():
    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return yaml.safe_load(file)


# 검출기 하나로 영상의 장면 경계 프레임 번호를 찾습니다. 첫 장면의 시작(0)은 경계가 아니라 제외합니다.
def find_boundaries(path, detector):
    from scenedetect import SceneManager, open_video

    manager = SceneManager()
    manager.add_detector(detector)
    manager.detect_scenes(open_video(str(path)), show_progress=False)
    return [scene[0].frame_num for scene in manager.get_scene_list()[1:]]


# 자식 프로세스가 맡는 일: 클립 하나를 여러 설정으로 검출합니다.
def detect_clip(row):
    from scenedetect import ContentDetector, ThresholdDetector

    record = {key: row[key] for key in ("clip_id", "source", "label", "kind", "length", "window_start", "window_end")}
    started = time.time()
    try:
        for threshold in CONTENT_THRESHOLDS:
            found = find_boundaries(row["path"], ContentDetector(threshold=threshold))
            record[f"content{threshold:g}"] = " ".join(str(frame) for frame in found)
        found = find_boundaries(row["path"], ThresholdDetector(threshold=12.0))
        record["fade"] = " ".join(str(frame) for frame in found)
        record["error"] = ""
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    record["seconds"] = round(time.time() - started, 1)
    return record


# 전환 구간 앞뒤 tolerance 안에 경계가 있으면 (찾았다, 위치 오차) 를 돌려줍니다.
def matches(boundaries, record, center):
    start, end = int(record["window_start"]), int(record["window_end"])
    if start == end:  # 하드컷
        start = end = center
    inside = [frame for frame in boundaries if start - TOLERANCE <= frame <= end + TOLERANCE]
    if not inside:
        return False, float("nan")
    # 전환 구간 중심에서 얼마나 떨어졌는지
    middle = (start + end) / 2
    return True, min(abs(frame - middle) for frame in inside)


def parse(text):
    return [int(value) for value in text.split()] if text else []


def main():
    parser = argparse.ArgumentParser(description="샷 경계 검출기가 합성 전환을 얼마나 잡는지 측정합니다.")
    parser.add_argument("--workers", type=int, default=3, help="동시에 돌릴 프로세스 수")
    parser.add_argument("--limit", type=int, default=None, help="앞에서부터 이 개수만 검사합니다(시험용).")
    args = parser.parse_args()

    config = load_config()
    center = config["benchmark"]["transition_center"]
    root = Path(config["benchmark"]["output_dir"])
    with open(root / "manifest.csv", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if args.limit:
        rows = rows[:args.limit]

    print(f"클립 {len(rows)}개를 검사합니다 (프로세스 {args.workers}개, 임계값 {CONTENT_THRESHOLDS} + 밝기 기반).")
    started = time.time()
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(detect_clip, row) for row in rows]
        for future in tqdm(as_completed(futures), total=len(futures), unit="클립"):
            records.append(future.result())
    failed = [record for record in records if record["error"]]
    print(f"완료 {len(records) - len(failed)}개 / 실패 {len(failed)}개 | 전체 {(time.time() - started) / 60:.1f}분")
    for record in failed[:3]:
        print(f"  실패: {record['clip_id']} - {record['error']}")

    methods = [f"content{threshold:g}" for threshold in CONTENT_THRESHOLDS] + ["fade", "합침"]
    by_kind = defaultdict(list)
    for record in records:
        if not record["error"]:
            kind = record["kind"] + (record["length"] if record["length"] != "0" else "")
            by_kind[kind].append(record)

    print(f"\n[검출률] 전환 구간 앞뒤 {TOLERANCE}프레임 안에서 경계를 찾은 클립 비율")
    print(f"{'전환':<12}{'개수':>5}" + "".join(f"{name:>12}" for name in methods))
    for kind in TRANSITION_KINDS:
        group = by_kind.get(kind, [])
        if not group:
            continue
        rates = []
        for method in methods:
            if method == "합침":
                found = [any(matches(parse(record[name]), record, center)[0]
                             for name in [f"content{CONTENT_THRESHOLDS[0]:g}", "fade"]) for record in group]
            else:
                found = [matches(parse(record[method]), record, center)[0] for record in group]
            rates.append(np.mean(found))
        print(f"{kind:<12}{len(group):>5}" + "".join(f"{rate:>11.0%} " for rate in rates))

    controls = by_kind.get("control", [])
    if controls:
        print(f"\n[오검출] 전환이 없는 control 클립 {len(controls)}개에서 경계가 잡힌 비율")
        print("   " + "  ".join(
            f"{method} {np.mean([len(parse(record[method])) > 0 for record in controls]):.0%}"
            for method in methods[:-1]))

    print("\n[위치 오차] 찾은 경우, 전환 중심에서 떨어진 프레임 수 중앙값 (기본 임계값 27 + 밝기 기반)")
    for kind in TRANSITION_KINDS:
        group = by_kind.get(kind, [])
        if not group:
            continue
        errors = []
        for record in group:
            boundaries = parse(record[f"content{CONTENT_THRESHOLDS[0]:g}"]) + parse(record["fade"])
            found, distance = matches(boundaries, record, center)
            if found:
                errors.append(distance)
        if errors:
            print(f"  {kind:<12} {np.median(errors):.0f}프레임 (찾은 클립 {len(errors)}개)")

    out_path = Path(config["paths"]["results"]) / "benchmark" / "shot_detection.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"\n클립별 결과 저장: {out_path}")


# 병렬 처리(자식 프로세스)가 있는 코드는 윈도우에서 반드시 이 조건 안에서 시작해야 합니다.
if __name__ == "__main__":
    main()
