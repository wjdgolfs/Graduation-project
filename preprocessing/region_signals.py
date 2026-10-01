# ============================================================================
# 2단계: 부위별 시간 변화량(D_region)과 배경 변화량(D_bg) 계산
# ----------------------------------------------------------------------------
# 무엇을 하나
#   얼굴 전처리 결과(preprocessing/extract_faces.py 의 .npz)를 읽어, 영상마다 프레임 t-1 → t 사이의
#     D_region[t, r] : 부위 r 안에서 얼굴 크롭(흑백) 픽셀이 얼마나 바뀌었나 (0~1)
#     D_bg[t]        : 배경(얼굴 주변을 넓게 뺀 전체 프레임)이 얼마나 바뀌었나 (0~1, 전처리 때 계산해 둔 값)
#   를 나란히 저장합니다. 5단계에서 R_r = D_region - α·D_bg 처럼 배경 변화로 부위 변화를 보정할 때의 입력입니다.
#   장면 전환(디졸브 등)은 배경과 얼굴을 함께 바꾸고, 합성 흔적은 얼굴 부위만 바꾼다는 가정을 확인하려는 것입니다.
#
# 부위 (얼굴 크롭 240x240 좌표, 괄호 안은 configs/config.yaml 의 regions 설정 이름)
#   eyes     : 양쪽 눈 윤곽의 볼록 껍질을 각각 eye_margin 픽셀 넓혀 합친 것 (눈꺼풀까지 포함)
#   mouth    : 입술 윤곽(바깥 + 안쪽)의 볼록 껍질을 mouth_margin 픽셀 넓힌 것
#   boundary : 얼굴 윤곽선 양쪽으로 boundary_band 픽셀씩 두꺼운 띠 (face-swap 이 섞는 경계)
#   face     : 얼굴 윤곽 볼록 껍질 전체 (눈·입 포함, 얼굴 전체의 기준값)
#
# 계산 방법
#   1) 랜드마크를 시간 방향으로 평균해 마스크 떨림을 줄입니다(landmark_smoothing, 이유는 face_regions.smooth_landmarks).
#   2) 프레임마다 부위 마스크를 만들고, 크롭 중 원래 프레임 밖(검정으로 채운 곳)은 뺍니다.
#   3) t-1 과 t 두 프레임 모두에서 부위 안인 픽셀(교집합)만 흑백 픽셀 차이 절댓값을 평균내고 255 로 나눕니다.
#      배경 변화량도 "두 프레임 모두 배경인 픽셀"만 비교했으므로 같은 규칙입니다.
#      크롭은 부드럽게 만든 상자로 잘랐기 때문에 머리 전체의 이동은 상당 부분 빠져 있고,
#      남는 값은 부위 안의 변화(표정·말하기 같은 실제 움직임 + 합성 흔적 + 압축 잡음)입니다.
#   4) landmark_motion[t, r]: 부위 랜드마크가 t-1 → t 에 평균 몇 픽셀 움직였는지.
#      말할 때 입 변화량이 커지는 것처럼, 실제 움직임 때문인 변화를 나중에 구분하는 데 씁니다.
#
# 입력: preprocess.output_dir/<데이터셋>/index.csv 와 영상별 얼굴 .npz
# 출력: paths.regions/<데이터셋>/<source>/<이름>.npz   영상 하나의 시계열
#       paths.regions/<데이터셋>/summary.csv            영상마다 시간 평균 한 줄 (라벨·분할 포함)
#
# .npz 안의 값 (F = 프레임 수, R = 부위 수. 시간 축의 t 번째 값은 "t-1 → t" 변화이고 t = 0 은 nan)
#   regions          [R] str        부위 이름 순서 (eyes, mouth, boundary, face)
#   region_diff      [F, R] float32 부위 변화량 D_region
#   region_area      [F, R] int32   프레임 t 의 부위 마스크 픽셀 수
#   landmark_motion  [F, R] float32 부위 랜드마크 평균 이동 거리(크롭 픽셀)
#   background_diff  [F] float32    배경 변화량 D_bg (얼굴 전처리 값 그대로)
#   background_ratio [F] float32    배경 마스크가 프레임에서 차지하는 비율 (작으면 D_bg 를 믿기 어려움)
#   frame_indices, label, video_id, settings(계산에 쓴 설정 JSON), error(성공이면 빈 문자열)
#
# 실행 예 (프로젝트 폴더에서)
#   python preprocessing/region_signals.py --dataset celebdf --limit 3     폴더마다 3개만 시험
#   python preprocessing/region_signals.py --dataset celebdf               전체 (이미 계산한 영상은 건너뜀)
#   python preprocessing/region_signals.py --dataset celebdf --overwrite   설정을 바꾼 뒤 전체 다시 계산
# ============================================================================

import argparse
import csv
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm


# __file__는 현재 파일 경로입니다. 상위 두 단계인 프로젝트 폴더를 기준으로 설정 파일을 찾습니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# 병렬 처리용 자식 프로세스도 이 파일을 다시 읽으므로 같은 경로가 들어갑니다.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.extract_faces import is_complete, limit_per_source  # noqa: E402
from preprocessing.face_regions import (  # noqa: E402
    boundary_band_mask,
    hull_mask,
    inside_frame_mask,
    masked_mean_abs_diff,
    region_point_indices,
    smooth_landmarks,
)


REGION_NAMES = ("eyes", "mouth", "boundary", "face")
# 얼굴 전처리 .npz 에서 읽는 값
FACE_KEYS = ("faces", "landmarks", "crop_boxes", "width", "height", "background_diff", "background_ratio", "frame_indices")
SUMMARY_COLUMNS = [
    "video_id", "source", "label", "split", "split_official", "frames", "valid_transitions",
    *[f"d_{name}" for name in REGION_NAMES],
    *[f"motion_{name}" for name in REGION_NAMES],
    "d_bg", "background_ratio", "background_ok", "error",
]

# 자식 프로세스마다 한 번 받아 두는 값(부위 점 번호, 설정)
_WORKER_STATE = {}


# 여러 스크립트가 같은 경로와 기준값을 쓰도록 YAML 설정을 딕셔너리로 읽습니다.
def load_config():
    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return yaml.safe_load(file)


# 프레임 하나의 부위 마스크 {부위 이름: [size, size] bool}. points 는 크롭 좌표 랜드마크 [478, 2] 입니다.
#   indices 는 face_regions.region_point_indices() 의 결과 {"face_oval", "left_eye", "right_eye", "lips": 점 번호 목록}
def region_masks(points, indices, size, settings):
    # 두 눈을 한 껍질로 감싸면 콧등까지 들어가므로, 눈마다 따로 만든 뒤 합칩니다(| 는 픽셀별 OR).
    eyes = (
        hull_mask(points, indices["left_eye"], size, size, settings["eye_margin"])
        | hull_mask(points, indices["right_eye"], size, size, settings["eye_margin"])
    )
    return {
        "eyes": eyes.astype(bool),
        "mouth": hull_mask(points, indices["lips"], size, size, settings["mouth_margin"]).astype(bool),
        "boundary": boundary_band_mask(points, indices["face_oval"], size, size, settings["boundary_band"]).astype(bool),
        "face": hull_mask(points, indices["face_oval"], size, size).astype(bool),
    }


# 부위마다 움직임을 잴 랜드마크 점 번호
def motion_indices(indices):
    return {
        "eyes": list(indices["left_eye"]) + list(indices["right_eye"]),
        "mouth": list(indices["lips"]),
        "boundary": list(indices["face_oval"]),
        "face": list(indices["face_oval"]),
    }


# 영상 하나의 부위 신호를 계산합니다. data 는 얼굴 전처리 .npz 에서 읽은 FACE_KEYS 값들의 딕셔너리입니다.
def compute_signals(data, indices, settings):
    faces = np.asarray(data["faces"])
    frames, size = faces.shape[0], faces.shape[1]
    width, height = int(data["width"]), int(data["height"])

    # 1) 흑백 크롭과 시간 방향으로 평균한 랜드마크
    gray = np.stack([cv2.cvtColor(face, cv2.COLOR_RGB2GRAY) for face in faces])
    points = smooth_landmarks(data["landmarks"], settings["landmark_smoothing"])

    # 2) 프레임마다 부위 마스크. 크롭 중 원래 프레임 밖(검정으로 채운 곳)은 뺍니다.
    masks = []
    for t in range(frames):
        inside = inside_frame_mask(data["crop_boxes"][t], width, height, size)
        masks.append({name: mask & inside for name, mask in region_masks(points[t], indices, size, settings).items()})

    # 3) t-1 → t 변화량과 랜드마크 움직임. 첫 프레임(t = 0)은 비교할 앞 프레임이 없어 nan 으로 둡니다.
    count = len(REGION_NAMES)
    region_diff = np.full((frames, count), np.nan, dtype=np.float32)
    region_area = np.array([[masks[t][name].sum() for name in REGION_NAMES] for t in range(frames)], dtype=np.int32)
    landmark_motion = np.full((frames, count), np.nan, dtype=np.float32)
    moving = motion_indices(indices)
    for t in range(1, frames):
        for r, name in enumerate(REGION_NAMES):
            # 두 프레임 모두에서 부위 안인 픽셀만 비교합니다. 너무 적으면(눈을 꼭 감았을 때 등) 값을 믿기 어려워 nan 으로 둡니다.
            both = masks[t - 1][name] & masks[t][name]
            if both.sum() >= settings["min_region_pixels"]:
                region_diff[t, r] = masked_mean_abs_diff(gray[t - 1], gray[t], both)
            # 점마다 이동 거리 √(Δx² + Δy²) 를 구해 평균냅니다.
            step = np.linalg.norm(points[t, moving[name]] - points[t - 1, moving[name]], axis=1)
            if np.isfinite(step).any():
                landmark_motion[t, r] = np.nanmean(step)

    return {
        "regions": np.array(REGION_NAMES),
        "region_diff": region_diff,
        "region_area": region_area,
        "landmark_motion": landmark_motion,
        "background_diff": np.asarray(data["background_diff"], dtype=np.float32),
        "background_ratio": np.asarray(data["background_ratio"], dtype=np.float32),
        "frame_indices": np.asarray(data["frame_indices"], dtype=np.int64),
    }


# 자식 프로세스가 시작될 때 한 번 실행됩니다.
def init_worker(indices, settings):
    # 프로세스 여러 개가 동시에 도는데 OpenCV 가 프로세스마다 스레드를 또 여러 개 만들면 오히려 느려져서 1개로 둡니다.
    cv2.setNumThreads(1)
    _WORKER_STATE["indices"] = indices
    _WORKER_STATE["settings"] = settings


# 자식 프로세스에서 실행되는 작업 하나: 영상 하나를 계산해 저장하고, 로그 한 줄을 돌려줍니다.
# 실패해도 멈추지 않고 오류 내용을 .npz 에 적어 둡니다(얼굴 전처리가 실패한 영상 등).
def process_video(job):
    row, faces_path, out_path = job
    settings = _WORKER_STATE["settings"]
    started = time.time()
    try:
        with np.load(faces_path) as data:
            face_error = str(data["error"]).strip()
            if face_error:
                raise RuntimeError(f"얼굴 전처리가 실패한 영상입니다: {face_error.splitlines()[-1]}")
            fields = {key: data[key] for key in FACE_KEYS}
        result = compute_signals(fields, _WORKER_STATE["indices"], settings)
        result["error"] = np.array("")
    except Exception:
        result = {"error": np.array(traceback.format_exc(limit=4))}
    result["video_id"] = np.array(row["video_id"])
    result["label"] = np.int8(int(row["label"]))
    # 어떤 설정으로 계산했는지 파일에 같이 남겨 둡니다(설정을 바꾼 뒤 옛 결과와 섞이지 않게 확인용).
    result["settings"] = np.array(json.dumps(settings, ensure_ascii=False))

    # 임시 이름으로 저장한 뒤 이름을 바꿉니다. 저장 도중에 끊기면 반쯤 쓴 파일이 완성본으로 남지 않습니다.
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(out_path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **result)
    os.replace(temporary, out_path)

    error = str(result["error"]).strip()
    return {"video_id": row["video_id"], "seconds": round(time.time() - started, 2), "error": error.splitlines()[-1] if error else ""}


# nan 을 뺀 평균. 쓸 값이 없으면 nan 입니다(np.nanmean 은 이때 경고를 냅니다).
def mean_or_nan(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else float("nan")


# 영상마다 시간 평균값을 한 줄씩 summary.csv 로 모읍니다. 목록(index.csv)의 모든 영상을 넣고, 결과가 없으면 error 에 적습니다.
def write_summary(rows, out_root, min_background_ratio):
    records = []
    for row in rows:
        record = {key: row[key] for key in ("video_id", "source", "label", "split", "split_official")}
        path = out_root / f"{row['video_id']}.npz"
        if not is_complete(path):
            record["error"] = "결과 없음"
            records.append(record)
            continue
        with np.load(path) as data:
            error = str(data["error"]).strip()
            if error:
                record["error"] = error.splitlines()[-1]
            else:
                diff = data["region_diff"][1:]
                motion = data["landmark_motion"][1:]
                record["frames"] = len(data["region_diff"])
                record["valid_transitions"] = int(np.isfinite(diff).all(axis=1).sum())
                for r, name in enumerate(REGION_NAMES):
                    record[f"d_{name}"] = round(mean_or_nan(diff[:, r]), 6)
                    record[f"motion_{name}"] = round(mean_or_nan(motion[:, r]), 4)
                record["d_bg"] = round(mean_or_nan(data["background_diff"][1:]), 6)
                record["background_ratio"] = round(float(np.mean(data["background_ratio"])), 4)
                record["background_ok"] = int(record["background_ratio"] >= min_background_ratio)
                record["error"] = ""
        records.append(record)

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "summary.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    return records


def main():
    parser = argparse.ArgumentParser(description="얼굴 전처리 결과로 부위별 시간 변화량과 배경 변화량을 계산합니다.")
    parser.add_argument("--dataset", required=True, choices=["celebdf", "ffpp"])
    parser.add_argument("--workers", type=int, default=None, help="동시에 돌릴 프로세스 수 (기본: 설정값)")
    parser.add_argument("--limit", type=int, default=None, help="폴더마다 앞에서부터 이 개수만 처리합니다(시험용).")
    parser.add_argument("--only", nargs="+", default=None, help="이 video_id 들만 처리합니다.")
    parser.add_argument("--overwrite", action="store_true", help="결과가 있어도 다시 계산합니다(설정을 바꿨을 때).")
    args = parser.parse_args()

    config = load_config()
    settings = config["regions"]
    workers = args.workers or settings["workers"]
    faces_root = Path(config["preprocess"]["output_dir"]) / args.dataset
    out_root = Path(config["paths"]["regions"]) / args.dataset

    # 1) 영상 목록: 얼굴 전처리가 만든 index.csv 를 그대로 씁니다.
    with open(faces_root / "index.csv", "r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    selected = rows
    if args.only:
        wanted = set(args.only)
        selected = [row for row in selected if row["video_id"] in wanted]
    if args.limit:
        selected = limit_per_source(selected, args.limit)

    # 2) 부위 점 번호는 mediapipe 가 있어야 구할 수 있어서, 메인 프로세스에서 한 번만 구해 자식 프로세스에 넘깁니다.
    indices = region_point_indices()
    jobs = [
        (row, str(faces_root / f"{row['video_id']}.npz"), str(out_root / f"{row['video_id']}.npz"))
        for row in selected
        if args.overwrite or not is_complete(out_root / f"{row['video_id']}.npz")
    ]
    print(f"[{args.dataset}] 영상 {len(selected)}개 중 {len(jobs)}개를 계산합니다 (프로세스 {workers}개).")

    if jobs:
        started = time.time()
        logs = []
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(indices, settings)) as pool:
            futures = {pool.submit(process_video, job): job[0] for job in jobs}
            for future in tqdm(as_completed(futures), total=len(futures), unit="영상"):
                try:
                    logs.append(future.result())
                except Exception as error:
                    # 자식 프로세스가 통째로 죽은 경우입니다. 결과 파일이 없으니 다시 실행하면 이 영상만 계산합니다.
                    logs.append({"video_id": futures[future]["video_id"], "seconds": 0.0, "error": f"{type(error).__name__}: {error}"})
        failed = [log for log in logs if log["error"]]
        print(f"완료 {len(logs) - len(failed)}개 / 실패 {len(failed)}개 | 전체 {(time.time() - started) / 60:.1f}분")
        for log in failed[:5]:
            print(f"  실패: {log['video_id']} - {log['error']}")

    # 3) 요약표는 이번에 계산하지 않은 영상까지 목록 전체로 다시 만듭니다.
    records = write_summary(rows, out_root, settings["min_background_ratio"])
    ok = [record for record in records if not record["error"]]
    print(f"요약: {out_root / 'summary.csv'} | 정상 {len(ok)}개 / 목록 {len(records)}개")
    if ok:
        for name in REGION_NAMES:
            values = np.array([record[f"d_{name}"] for record in ok], dtype=float)
            print(f"  D_{name:<9} 영상 평균의 중앙값 {np.nanmedian(values):.4f}")
        low = sum(record["background_ok"] == 0 for record in ok)
        print(f"  D_bg       영상 평균의 중앙값 {np.nanmedian([record['d_bg'] for record in ok]):.4f} | "
              f"배경 면적 비율 {settings['min_background_ratio']} 미만 영상 {low}개")


# 병렬 처리(자식 프로세스)가 있는 코드는 윈도우에서 반드시 이 조건 안에서 시작해야 합니다.
if __name__ == "__main__":
    main()
