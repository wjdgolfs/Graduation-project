# ============================================================================
# 3단계 벤치마크 클립의 부위·배경 신호 계산
# ----------------------------------------------------------------------------
# 무엇을 하나
#   전환 벤치마크 클립(preprocessing/build_transition_benchmark.py 가 만든 것)마다
#   전환 구간을 가운데 두고 analysis_frames(128)장을 골라, 프레임마다
#     D_region[t, r] : 눈·입·얼굴 경계·얼굴 전체의 변화량
#     D_bg[t]        : 배경(얼굴 주변을 넓게 뺀 화면)의 변화량
#   을 계산합니다. 5단계 정규화의 핵심 질문에 답하기 위한 데이터입니다.
#
#   "장면 전환이 만든 변화는 배경에도 똑같이 나타난다. 그러므로 D_bg 로 정규화하면
#    전환 때문에 생긴 부위 변화는 사라지고, 합성 얼굴 때문에 생긴 변화만 남아야 한다."
#
# 계산 구조
#   얼굴 전처리(extract_faces.extract_video)와 부위 신호(region_signals.compute_signals)를
#   그대로 불러 씁니다. 같은 계산을 두 번 구현하지 않기 위해서입니다.
#
# 왜 얼굴 크롭은 저장하지 않나
#   클립 1,260개 x 128프레임 크롭을 모두 저장하면 20GB 가 넘습니다.
#   신호만 저장하면 클립당 수십 KB 라, 전환 구간 분석에 필요한 것만 남습니다.
#
# 입력: benchmark.output_dir/manifest.csv 와 그 안의 클립 mp4
# 출력: benchmark.output_dir/signals/<기법>/<클립>.npz
#
# .npz 안의 값 (F = analysis_frames, R = 부위 수. t 번째 값은 "t-1 → t" 변화, t = 0 은 nan)
#   regions, region_diff, region_area, landmark_motion, background_diff, background_ratio, face_found
#   frame_indices : 원본 클립에서 몇 번 프레임인지 (전환 구간과 맞춰 보는 데 씁니다)
#   clip_id, source, kind, length, label, window_start, window_end, error
#
# 실행 예 (프로젝트 폴더에서)
#   python preprocessing/benchmark_signals.py --limit 12   시험
#   python preprocessing/benchmark_signals.py              전체
# ============================================================================

import argparse
import csv
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm


# __file__는 현재 파일 경로입니다. 상위 두 단계인 프로젝트 폴더를 기준으로 설정 파일을 찾습니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# 병렬 처리용 자식 프로세스도 이 파일을 다시 읽으므로 같은 경로가 들어갑니다.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.extract_faces import extract_video, init_worker as init_face_worker, is_complete  # noqa: E402
from preprocessing.face_regions import region_point_indices  # noqa: E402
from preprocessing.region_signals import compute_signals  # noqa: E402


# 자식 프로세스마다 한 번 받아 두는 값
_WORKER_STATE = {}


# 여러 스크립트가 같은 경로와 기준값을 쓰도록 YAML 설정을 딕셔너리로 읽습니다.
def load_config():
    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return yaml.safe_load(file)


# 얼굴 전처리 설정을 벤치마크 분석용으로 바꿉니다.
# 전환 중심을 가운데 두고 analysis_frames 장을 연속으로 씁니다.
#   예) 전환 중심 128, 분석 128장 → 64번부터 191번 프레임 (가장 긴 64프레임 디졸브 96~159 를 여유 있게 덮습니다)
def analysis_settings(config):
    benchmark = config["benchmark"]
    settings = dict(config["preprocess"])
    settings["frames"] = benchmark["analysis_frames"]
    settings["sampling"] = "consecutive"
    settings["start_frame"] = benchmark["transition_center"] - benchmark["analysis_frames"] // 2
    return settings


def init_worker(model_path, indices, face_settings, region_settings):
    # MediaPipe 모델을 읽고 C++ 로그를 끕니다(extract_faces 와 같은 준비).
    init_face_worker(model_path)
    _WORKER_STATE["indices"] = indices
    _WORKER_STATE["face"] = face_settings
    _WORKER_STATE["region"] = region_settings


# 자식 프로세스가 맡는 일: 클립 하나의 신호를 계산해 저장합니다.
def process_clip(job):
    row, out_path = job
    started = time.time()
    try:
        faces = extract_video({"path": row["path"]}, _WORKER_STATE["face"])
        result = dict(compute_signals(faces, _WORKER_STATE["indices"], _WORKER_STATE["region"]))
        result["face_found"] = faces["face_found"]
        result["error"] = np.array("")
    except Exception:
        result = {"error": np.array(traceback.format_exc(limit=4))}

    for key in ("clip_id", "source", "kind"):
        result[key] = np.array(row[key])
    for key in ("label", "length", "window_start", "window_end"):
        result[key] = np.int32(int(row[key]))

    # 임시 이름으로 저장한 뒤 이름을 바꿉니다. 저장 도중 끊겨도 반쯤 쓴 파일이 완성본으로 남지 않습니다.
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(out_path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **result)
    os.replace(temporary, out_path)

    error = str(result["error"]).strip()
    return {
        "clip_id": row["clip_id"],
        "found": int(result["face_found"].mean() * 100) if "face_found" in result else 0,
        "seconds": round(time.time() - started, 1),
        "error": error.splitlines()[-1] if error else "",
    }


def main():
    parser = argparse.ArgumentParser(description="전환 벤치마크 클립의 부위·배경 신호를 계산합니다.")
    parser.add_argument("--workers", type=int, default=None, help="동시에 돌릴 프로세스 수 (기본: 설정값)")
    parser.add_argument("--limit", type=int, default=None, help="앞에서부터 이 개수만 계산합니다(시험용).")
    parser.add_argument("--overwrite", action="store_true", help="결과가 있어도 다시 계산합니다.")
    args = parser.parse_args()

    config = load_config()
    benchmark = config["benchmark"]
    workers = args.workers or benchmark["workers"]
    root = Path(benchmark["output_dir"])
    out_root = root / "signals"

    with open(root / "manifest.csv", "r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if args.limit:
        rows = rows[:args.limit]

    jobs = []
    for row in rows:
        out_path = out_root / row["source"] / f"{row['clip_id']}.npz"
        if args.overwrite or not is_complete(out_path):
            jobs.append((row, str(out_path)))
    print(f"클립 {len(rows)}개 중 {len(jobs)}개를 계산합니다 (프로세스 {workers}개).")
    if not jobs:
        return

    face_settings = analysis_settings(config)
    print(f"분석 구간: {face_settings['start_frame']}번부터 {face_settings['frames']}프레임 "
          f"(전환 중심 {benchmark['transition_center']})")

    started = time.time()
    logs = []
    initargs = (config["mediapipe"]["face_landmarker"], region_point_indices(), face_settings, config["regions"])
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=initargs) as pool:
        futures = [pool.submit(process_clip, job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), unit="클립"):
            logs.append(future.result())

    failed = [log for log in logs if log["error"]]
    elapsed = (time.time() - started) / 60
    print(f"완료 {len(logs) - len(failed)}개 / 실패 {len(failed)}개 | 전체 {elapsed:.1f}분")
    for log in failed[:5]:
        print(f"  실패: {log['clip_id']} - {log['error']}")
    if len(logs) > len(failed):
        found = np.mean([log["found"] for log in logs if not log["error"]])
        print(f"얼굴 찾은 프레임 비율 평균 {found:.1f}% | 저장 위치: {out_root}")


# 병렬 처리(자식 프로세스)가 있는 코드는 윈도우에서 반드시 이 조건 안에서 시작해야 합니다.
if __name__ == "__main__":
    main()
