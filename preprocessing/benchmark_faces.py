# ============================================================================
# 벤치마크 클립의 얼굴 크롭 추출 (원 논문 모델로 채점하기 위한 입력)
# ----------------------------------------------------------------------------
# 무엇을 하나
#   전환 벤치마크 클립(preprocessing/build_transition_benchmark.py 결과)마다
#   전환 중심을 가운데 둔 연속 30프레임을 골라 얼굴 크롭·랜드마크·배경 통계를 저장합니다.
#   원 논문 모델(models/paper_bclstm.py)과 5단계 제안 모델의 입력이 이 형식입니다.
#
# 왜 30프레임인가
#   원 논문 모델의 입력이 30프레임(240x240)입니다. 전환 중심 128 기준으로 113~142번 프레임을 씁니다.
#   FTCN(32프레임 클립을 겹쳐 가며 영상 전체를 보는 방식)과 보는 범위가 달라서,
#   두 모델의 점수를 직접 비교하지 않고 각각 "대조군 대비 변화량"으로 비교합니다.
#   64프레임 디졸브(96~159)는 이 윈도 전체가 전환 구간 안에 들어갑니다. 해석할 때 유의하세요.
#
# 신호(benchmark_signals.py)와 따로 두는 이유
#   신호 계산은 128프레임을 보지만 크롭을 저장하지 않습니다(용량). 모델 채점은 크롭이 필요합니다.
#   계산은 둘 다 extract_faces.extract_video 를 그대로 불러 씁니다.
#
# 용량: 클립 하나당 약 3~5MB (30 x 240 x 240 x 3). 210클립 묶음 약 1GB, 1,260클립 전체 약 6GB.
#
# 입력: --manifest (기본 benchmark.output_dir/manifest.csv) 와 그 안의 클립 mp4
# 출력: benchmark.output_dir/faces/<기법>/<클립>.npz
#       extract_faces 와 같은 항목(faces, landmarks, thumbs, background_diff ...)에
#       clip_id, source, kind, label, length, window_start, window_end 를 더해 저장합니다.
#
# 실행 예 (프로젝트 폴더에서)
#   python preprocessing/benchmark_faces.py --manifest D:/Graduation-project-data/processed/benchmark/manifest_pass1.csv
#   python preprocessing/benchmark_faces.py            전체 1,260클립
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.extract_faces import extract_video, init_worker as init_face_worker, is_complete  # noqa: E402


_WORKER_STATE = {}

# 저장할 때 숫자로 바꿀 열과 문자열로 둘 열
TEXT_COLUMNS = ("clip_id", "source", "kind")
INT_COLUMNS = ("label", "length", "window_start", "window_end")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# 얼굴 전처리 설정을 "모델 입력" 용으로 바꿉니다: 전환 중심을 가운데 둔 연속 30프레임.
def window_settings(config):
    benchmark = config["benchmark"]
    paper = config["paper_model"]
    settings = dict(config["preprocess"])
    settings["frames"] = paper["frames"]
    settings["crop_size"] = paper["image_size"]
    settings["sampling"] = "consecutive"
    settings["start_frame"] = benchmark["transition_center"] - paper["frames"] // 2
    return settings


def init_worker(model_path, face_settings):
    init_face_worker(model_path)
    _WORKER_STATE["face"] = face_settings


# 자식 프로세스가 맡는 일: 클립 하나의 얼굴 크롭을 뽑아 저장합니다.
def process_clip(job):
    row, out_path = job
    started = time.time()
    try:
        result = dict(extract_video({"path": row["path"]}, _WORKER_STATE["face"]))
        result["error"] = np.array("")
    except Exception:
        result = {"error": np.array(traceback.format_exc(limit=4))}

    for key in TEXT_COLUMNS:
        result[key] = np.array(row[key])
    for key in INT_COLUMNS:
        result[key] = np.int32(int(row[key]))

    # 임시 이름으로 저장한 뒤 바꿉니다. 저장 도중 끊겨도 반쯤 쓴 파일이 완성본으로 남지 않습니다.
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(out_path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **result)
    os.replace(temporary, out_path)

    error = str(result["error"]).strip()
    return {
        "clip_id": row["clip_id"],
        "found": float(result["face_found"].mean() * 100) if "face_found" in result else 0.0,
        "seconds": round(time.time() - started, 1),
        "error": error.splitlines()[-1] if error else "",
    }


def main():
    config = load_config()
    benchmark = config["benchmark"]
    root = Path(benchmark["output_dir"])

    parser = argparse.ArgumentParser(description="벤치마크 클립에서 전환 중심 30프레임 얼굴 크롭을 뽑습니다.")
    parser.add_argument("--manifest", default=str(root / "manifest.csv"), help="처리할 클립 목록 CSV")
    parser.add_argument("--workers", type=int, default=None, help="동시에 돌릴 프로세스 수 (기본: 설정값)")
    parser.add_argument("--limit", type=int, default=None, help="앞에서부터 이 개수만 처리합니다(시험용).")
    parser.add_argument("--overwrite", action="store_true", help="결과가 있어도 다시 처리합니다.")
    args = parser.parse_args()

    workers = args.workers or benchmark["workers"]
    out_root = root / "faces"

    with open(args.manifest, "r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if args.limit:
        rows = rows[:args.limit]

    # clip_id 는 기법마다 겹칩니다(FF++ 는 같은 번호의 영상을 기법마다 가집니다). 그래서 기법별 하위 폴더에 저장합니다.
    jobs = []
    for row in rows:
        out_path = out_root / row["source"] / f"{row['clip_id']}.npz"
        if args.overwrite or not is_complete(out_path):
            jobs.append((row, str(out_path)))
    print(f"클립 {len(rows)}개 중 {len(jobs)}개를 처리합니다 (프로세스 {workers}개).")
    if not jobs:
        return

    face_settings = window_settings(config)
    print(f"입력 구간: {face_settings['start_frame']}번부터 {face_settings['frames']}프레임, "
          f"크롭 {face_settings['crop_size']}px (전환 중심 {benchmark['transition_center']})")

    started = time.time()
    logs = []
    initargs = (config["mediapipe"]["face_landmarker"], face_settings)
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


if __name__ == "__main__":
    main()
