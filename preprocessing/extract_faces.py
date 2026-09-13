# ============================================================================
# 얼굴 전처리: 영상 → 얼굴 크롭 + 478점 랜드마크 + 배경 통계 (.npz)
# ----------------------------------------------------------------------------
# 무엇을 하나
#   영상마다 프레임 frames(30)장을 골라, 프레임마다
#     1) MediaPipe 로 얼굴 478점을 찾고 (아래 "얼굴 찾는 방법")
#     2) 얼굴을 넉넉히 둘러싼 정사각형을 crop_size(240)로 잘라 저장하고
#     3) 랜드마크를 크롭 좌표로 옮겨 저장하고(나중에 눈·입·경계 마스크를 원하는 해상도로 만들 수 있게)
#     4) 전체 프레임의 흑백 축소판과 배경 마스크로 배경 변화량 D_bg 를 계산해 저장합니다.
#   원 논문 재현(얼굴 크롭 30프레임, 240x240)과 이 연구의 배경 정규화에 필요한 값을 한 번에 뽑습니다.
#   계획서의 주의점대로, 얼굴만 잘라 넘기면 배경 정보가 사라지므로 전처리 단계부터 배경 통계를 같이 뽑습니다.
#
# 얼굴 찾는 방법 (두 단계)
#   MediaPipe 얼굴 검출기는 얼굴이 화면을 크게 차지할 때 잘 동작합니다. Celeb-DF 는 얼굴 폭이 화면 폭의 10% 안팎이라
#   전체 프레임을 넣으면 놓치는 영상이 많았습니다(시험 6개 중 1개는 30프레임 모두 실패, 가운데를 잘라 넣으면 모두 성공).
#   그래서 프레임마다 다음 순서로 찾습니다.
#     1) 앞 프레임에서 얼굴을 찾았으면, 그 얼굴 상자를 search_window_scale(3)배로 넓힌 영역만 잘라 넣습니다.
#     2) 처음이거나 1)에서 못 찾으면, 프레임 전체 → 겹치는 창들(가운데 창부터)을 차례로 넣어 봅니다.
#   잘라 넣는 영역이 프레임마다 달라져서 MediaPipe 의 video 모드(자체 추적)는 쓰지 않고 image 모드로 찾습니다.
#   추적은 1)이 대신하고, 크롭 상자의 떨림은 이동 평균(box_smoothing)으로 줄입니다.
#
# 랜드마크 떨림과 시간 평활 (Celeb-DF 30개 영상으로 잰 값, 크롭 240 픽셀 기준)
#   저장하는 랜드마크는 프레임마다 따로 찾은 원래 값입니다. 떨림(점마다 2차 차분의 중앙값)은 1.87px 입니다.
#   시간 방향으로 부드럽게 만드는 일은 부위 마스크를 만드는 단계에서 고르도록 남겨 둡니다.
#     - 앞뒤 1프레임씩 3프레임 평균: 떨림 0.60px, 움직임 지연 없음. 다만 빠른 눈 깜빡임 폭이 줄어듭니다(18.4 → 10.3px).
#     - MediaPipe video 모드: 떨림 0.69px 이지만, 앞 프레임만 보고 거르기 때문에 움직이는 방향 뒤로 평균 0.6px 처집니다.
#   평활 방식을 바꿀 때마다 전처리를 처음부터 다시 돌리지 않도록, 원래 값을 저장합니다.
#
# 입력: config.yaml 의 raw 경로 아래 Celeb-DF-v2 또는 FaceForensics++
# 출력: preprocess.output_dir/<데이터셋>/index.csv            영상 목록·라벨·분할 (build_index.py)
#       preprocess.output_dir/<데이터셋>/<source>/<이름>.npz   영상 하나의 전처리 결과
#       preprocess.output_dir/<데이터셋>/extract_log.csv      이번 실행의 영상별 처리 결과
#
# 실행 예 (프로젝트 폴더에서)
#   python preprocessing/extract_faces.py --dataset celebdf --limit 3     폴더마다 3개씩만 시험
#   python preprocessing/extract_faces.py --dataset celebdf               전체 (이미 한 영상은 건너뜀)
#   python preprocessing/extract_faces.py --dataset ffpp --workers 6
#
# .npz 안의 값 (F = frames, S = crop_size)
#   faces            [F, S, S, 3] uint8   얼굴 크롭 (RGB)
#   landmarks        [F, 478, 2] float32  크롭 좌표의 랜드마크 (얼굴을 못 찾은 프레임은 nan)
#   face_found       [F] bool             얼굴을 찾았는지
#   frame_indices    [F] int64            원본 영상의 프레임 번호
#   face_boxes       [F, 4] float32       원본 좌표의 얼굴 상자(랜드마크를 감싼 상자, 못 찾으면 nan)
#   smoothed_boxes   [F, 4] float32       face_boxes 의 빈 프레임을 채우고 이동 평균한 상자
#   crop_boxes       [F, 4] float32       원본 좌표의 크롭 상자(smoothed_boxes 를 crop_scale 배 넓힌 정사각형)
#   thumbs           [F, h, w] uint8      전체 프레임의 흑백 축소판 (긴 변 thumb_max_side)
#   thumb_scale      float                축소판 크기 / 원본 크기
#   background_ratio [F] float32          축소판에서 배경 마스크가 차지하는 비율 (작으면 배경 통계를 믿기 어려움)
#   background_diff  [F] float32          배경의 프레임 간 변화량 D_bg (t-1 → t, 0~1). 첫 프레임은 nan
#   fps, width, height, label, video_id, error(성공이면 빈 문자열), elapsed_sec
#
# 병렬 처리
#   MediaPipe 는 CPU 로 돕니다(프레임당 약 16ms). 영상 여러 개를 프로세스 여러 개(workers)가 나눠 처리합니다.
#   프로세스는 스레드와 달리 메모리를 따로 쓰므로, 무거운 CPU 계산을 진짜로 동시에 할 수 있습니다.
# ============================================================================

import argparse
import csv
import os
import sys
import time
import traceback
import zipfile  # .npz 는 zip 파일이라, 끝까지 저장됐는지 zip 목차로 확인합니다.
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# MediaPipe 가 내부 C++ 로그를 많이 찍어서, glog 로그는 오류 이상만 보이게 합니다. mediapipe 를 import 하기 전에 정해야 합니다.
# 이 설정으로 막히지 않는 경고("Feedback manager requires ..." 등)는 init_worker 에서 따로 끕니다.
os.environ.setdefault("GLOG_minloglevel", "2")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from tqdm import tqdm  # noqa: E402


# __file__는 현재 파일 경로입니다. 상위 두 단계인 프로젝트 폴더를 기준으로 설정 파일을 찾습니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# 다른 폴더의 공용 코드를 불러오려면 프로젝트 폴더가 import 경로에 있어야 합니다.
# 병렬 처리용 자식 프로세스도 이 파일을 다시 읽으므로 같은 경로가 들어갑니다.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.build_index import build_rows, write_index  # noqa: E402
from preprocessing.face_regions import (  # noqa: E402
    background_mask,
    box_iou,
    clip_window,
    crop_frame,
    masked_mean_abs_diff,
    search_windows,
    smooth_boxes,
    square_box,
    tight_box,
    to_crop_coords,
)
from utils.video_io import probe_video, read_frames, read_selected_frames  # noqa: E402


LANDMARK_COUNT = 478
# 이보다 작은 창은 MediaPipe 에 넣지 않습니다(프레임 가장자리에서 잘려 거의 남지 않은 창).
MIN_WINDOW_SIDE = 16
LOG_COLUMNS = ["video_id", "label", "frames", "faces_found", "background_ratio_mean", "background_diff_mean", "seconds", "error"]

# 자식 프로세스마다 한 번 읽어 두는 모델 파일 내용. 프로세스끼리는 메모리를 공유하지 않아서 각자 가집니다.
_WORKER_STATE = {}


# 여러 스크립트가 같은 경로와 기준값을 쓰도록 YAML 설정을 딕셔너리로 읽습니다.
def load_config():
    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return yaml.safe_load(file)


# 자식 프로세스가 시작될 때 한 번 실행됩니다. 모델 파일을 읽어 두고, 필요하면 C++ 로그 출력을 끕니다.
def init_worker(model_path, native_logs=False):
    _WORKER_STATE["model"] = Path(model_path).read_bytes()
    if not native_logs:
        # MediaPipe 는 모델을 만들 때마다(영상마다) C++ 경고를 표준 오류로 직접 찍어서, 영상 수천 개를 돌리면 화면이 경고로 덮입니다.
        # 파일 번호 2(표준 오류)를 버리는 곳(os.devnull)으로 바꿔 끼웁니다(dup2). 이 자식 프로세스에만 적용됩니다.
        # 파이썬 오류는 process_video 가 잡아서 결과로 돌려주므로 메인 프로세스 화면과 로그에 그대로 남습니다.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.close(devnull)


# 영상에서 설정대로 프레임을 고릅니다. 반환값: (프레임 배열 [N, H, W, 3], fps, 프레임 번호 배열)
def load_frames(path, settings):
    count = settings["frames"]
    if settings["sampling"] == "consecutive":
        start = settings["start_frame"]
        frames, fps = read_frames(path, start=start, count=count)
        return frames, fps, np.arange(start, start + len(frames))
    if settings["sampling"] == "uniform":
        total = probe_video(path)["frames"]
        # linspace: 0 ~ 마지막 프레임을 count 개로 고르게 나눈 번호
        wanted = np.linspace(0, max(total - 1, 0), count).round().astype(int)
        return read_selected_frames(path, wanted)
    raise ValueError(f"알 수 없는 sampling 입니다: {settings['sampling']}")


# 한 프레임에서 얼굴이 여러 개 찾아졌을 때 하나를 고릅니다.
#   앞 프레임의 얼굴 상자가 있으면 그것과 가장 많이 겹치는 얼굴(같은 사람일 가능성이 큼),
#   없으면 가장 큰 얼굴을 고릅니다.
def choose_face(candidates, previous_box):
    boxes = [tight_box(points) for points in candidates]
    if previous_box is None or not np.isfinite(previous_box).all():
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
        return candidates[int(np.argmax(areas))]
    overlaps = [box_iou(box, previous_box) for box in boxes]
    return candidates[int(np.argmax(overlaps))]


# 프레임마다 MediaPipe 로 478점을 찾습니다. 반환값: [F, 478, 2] 원본 픽셀 좌표 (못 찾은 프레임은 nan)
def detect_landmarks(frames, settings):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

    height, width = frames.shape[1:3]
    options = vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_buffer=_WORKER_STATE["model"]),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=settings["max_faces"],
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    # 앞 프레임 얼굴이 없을 때 차례로 시도할 창: 프레임 전체, 그다음 겹치는 창들(가운데부터)
    fallback_windows = [(0, 0, width, height)] + search_windows(height, width, settings["search_window_fraction"])

    # 창 (x1, y1, x2, y2) 부분을 잘라 넣고, 찾은 얼굴들의 점을 원본 프레임 좌표로 돌려줍니다.
    def detect_in_window(frame, window):
        x1, y1, x2, y2 = window
        crop = np.ascontiguousarray(frame[y1:y2, x1:x2])
        crop_height, crop_width = crop.shape[:2]
        if crop_height < MIN_WINDOW_SIDE or crop_width < MIN_WINDOW_SIDE:
            return []
        detection = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
        # MediaPipe 좌표는 잘라 넣은 이미지 기준 0~1 이라, 창 크기를 곱하고 창의 왼쪽 위 좌표를 더해 원본 좌표로 바꿉니다.
        return [
            np.array([[x1 + landmark.x * crop_width, y1 + landmark.y * crop_height] for landmark in face])
            for face in detection.face_landmarks
            if len(face) == LANDMARK_COUNT
        ]

    points = np.full((len(frames), LANDMARK_COUNT, 2), np.nan, dtype=np.float64)
    previous_box = None
    try:
        for k, frame in enumerate(frames):
            candidates = []
            # 1) 앞 프레임 얼굴 주변만 잘라서 찾기
            if previous_box is not None:
                window = clip_window(square_box(previous_box, settings["search_window_scale"]), height, width)
                candidates = detect_in_window(frame, window)
            # 2) 못 찾았으면 프레임 전체, 겹치는 창 순서로 찾기
            if not candidates:
                for window in fallback_windows:
                    candidates = detect_in_window(frame, window)
                    if candidates:
                        break
            if not candidates:
                # 이 프레임은 비워 두고, 다음 프레임은 마지막으로 찾은 얼굴 주변부터 다시 찾습니다.
                continue
            points[k] = choose_face(candidates, previous_box)
            previous_box = tight_box(points[k])
    finally:
        landmarker.close()

    return points


# 영상 하나를 전처리해 .npz 에 넣을 값들을 돌려줍니다.
def extract_video(row, settings):
    frames, fps, frame_indices = load_frames(row["path"], settings)
    if len(frames) < settings["frames"]:
        raise RuntimeError(f"프레임이 {settings['frames']}장보다 적습니다: {len(frames)}장")
    height, width = frames.shape[1:3]

    # 1) 랜드마크
    points = detect_landmarks(frames, settings)

    # 2) 얼굴 상자 → 부드럽게 → 정사각형 크롭 상자
    face_boxes = np.stack([tight_box(frame_points) for frame_points in points])
    smoothed, found = smooth_boxes(face_boxes, settings["box_smoothing"])
    if not found.any():
        raise RuntimeError("어느 프레임에서도 얼굴을 찾지 못했습니다.")
    crop_boxes = square_box(smoothed, settings["crop_scale"])

    # 3) 얼굴 크롭과 크롭 좌표 랜드마크
    size = settings["crop_size"]
    faces = np.stack([crop_frame(frame, box, size) for frame, box in zip(frames, crop_boxes)])
    landmarks = np.stack([
        to_crop_coords(frame_points, box, size) for frame_points, box in zip(points, crop_boxes)
    ])

    # 4) 배경: 흑백 축소판에서 얼굴 주변을 넓게 뺀 영역의 프레임 간 변화량
    thumb_scale = settings["thumb_max_side"] / max(height, width)
    thumb_size = (max(1, int(round(width * thumb_scale))), max(1, int(round(height * thumb_scale))))
    thumbs = np.stack([
        cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), thumb_size, interpolation=cv2.INTER_AREA)
        for frame in frames
    ])
    thumb_height, thumb_width = thumbs.shape[1:3]
    # 얼굴을 못 찾은 프레임도 얼굴 자리를 빼도록, 채워서 부드럽게 만든 상자(smoothed)를 씁니다.
    masks = [
        background_mask(thumb_height, thumb_width, box * thumb_scale, settings["background_exclusion_scale"])
        for box in smoothed
    ]
    background_ratio = np.array([mask.mean() for mask in masks], dtype=np.float32)
    background_diff = np.full(len(frames), np.nan, dtype=np.float32)
    for k in range(1, len(frames)):
        # 두 프레임 모두에서 배경인 픽셀만 비교합니다.
        background_diff[k] = masked_mean_abs_diff(thumbs[k - 1], thumbs[k], masks[k - 1] & masks[k])

    return {
        "faces": faces,
        "landmarks": landmarks.astype(np.float32),
        "face_found": found,
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "face_boxes": face_boxes.astype(np.float32),
        "smoothed_boxes": smoothed.astype(np.float32),
        "crop_boxes": crop_boxes.astype(np.float32),
        "thumbs": thumbs,
        "thumb_scale": np.float32(thumb_scale),
        "background_ratio": background_ratio,
        "background_diff": background_diff,
        "fps": np.float32(fps),
        "width": np.int32(width),
        "height": np.int32(height),
    }


# 자식 프로세스에서 실행되는 작업 하나: 영상을 전처리해 저장하고, 로그 한 줄을 돌려줍니다.
# 실패해도 멈추지 않고 오류 내용을 .npz 에 적어 둡니다(다시 실행할 때 같은 영상을 반복 시도하지 않게).
def process_video(job):
    row, settings, out_path = job
    started = time.time()
    try:
        result = extract_video(row, settings)
        result["error"] = np.array("")
    except Exception:
        result = {"error": np.array(traceback.format_exc(limit=6))}
    result["video_id"] = np.array(row["video_id"])
    result["label"] = np.int8(int(row["label"]))
    result["elapsed_sec"] = np.float32(time.time() - started)

    # 임시 이름으로 저장한 뒤 이름을 바꿉니다. 저장 도중에 끊기면 반쯤 쓴 파일이 완성본으로 남지 않습니다.
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(out_path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **result)
    os.replace(temporary, out_path)

    error = str(result["error"]).strip()
    return {
        "video_id": row["video_id"],
        "label": row["label"],
        "frames": len(result.get("faces", [])),
        "faces_found": int(result["face_found"].sum()) if "face_found" in result else 0,
        "background_ratio_mean": float(np.mean(result["background_ratio"])) if "background_ratio" in result else float("nan"),
        "background_diff_mean": float(np.nanmean(result["background_diff"])) if "background_diff" in result else float("nan"),
        "seconds": round(float(result["elapsed_sec"]), 2),
        "error": error.splitlines()[-1] if error else "",
    }


# 결과 파일이 끝까지 저장됐는지 확인합니다. 반환값: 제대로 열리면 True (파일이 없어도 False)
# 저장 직후 PC 가 갑자기 꺼지면 크기는 정상인데 내용이 비어 열리지 않는 파일이 남을 수 있습니다
# (2026-09-13 블루스크린 때 YouTube-real 53개). zip 목차를 읽을 수 있고 error 항목이 있으면 완성본으로 봅니다.
# 목차만 읽어서 영상 수천 개도 몇 초면 끝납니다. 내용 중간만 망가진 경우까지 보려면 zipfile 의 testzip 으로 전체를 검사해야 합니다.
def is_complete(path):
    try:
        with zipfile.ZipFile(path) as archive:
            return "error.npy" in archive.namelist()
    except (OSError, zipfile.BadZipFile):
        return False


# 폴더(source)마다 앞에서부터 limit 개씩 고릅니다. 시험 삼아 돌릴 때 real 과 fake 가 모두 들어가게 합니다.
def limit_per_source(rows, limit):
    kept, counts = [], {}
    for row in rows:
        if counts.get(row["source"], 0) < limit:
            kept.append(row)
            counts[row["source"]] = counts.get(row["source"], 0) + 1
    return kept


def main():
    parser = argparse.ArgumentParser(description="영상에서 얼굴 크롭·랜드마크·배경 통계를 뽑습니다.")
    parser.add_argument("--dataset", required=True, choices=["celebdf", "ffpp"])
    parser.add_argument("--workers", type=int, default=None, help="동시에 돌릴 프로세스 수 (기본: 설정값)")
    parser.add_argument("--limit", type=int, default=None, help="폴더마다 앞에서부터 이 개수만 처리합니다(시험용).")
    parser.add_argument("--only", nargs="+", default=None, help="이 video_id 들만 처리합니다.")
    parser.add_argument("--overwrite", action="store_true", help="결과가 있어도 다시 처리합니다.")
    parser.add_argument("--native-logs", action="store_true", help="MediaPipe C++ 로그를 끄지 않고 보여줍니다(문제를 찾을 때).")
    args = parser.parse_args()

    config = load_config()
    settings = config["preprocess"]
    model_path = config["mediapipe"]["face_landmarker"]
    workers = args.workers or settings["workers"]

    # 1) 목록 만들기
    rows = build_rows(args.dataset, config)
    out_root = Path(settings["output_dir"]) / args.dataset
    write_index(rows, out_root / "index.csv")

    if args.only:
        wanted = set(args.only)
        rows = [row for row in rows if row["video_id"] in wanted]
    if args.limit:
        rows = limit_per_source(rows, args.limit)

    # 2) 아직 처리하지 않았거나 결과 파일이 망가진 영상만 고르기. video_id 의 "/" 는 폴더 구분이 됩니다.
    jobs = []
    for row in rows:
        out_path = out_root / f"{row['video_id']}.npz"
        if args.overwrite or not is_complete(out_path):
            jobs.append((row, settings, str(out_path)))
    print(f"[{args.dataset}] 영상 {len(rows)}개 중 {len(jobs)}개를 처리합니다 (프로세스 {workers}개).")
    if not jobs:
        return

    # 3) 병렬 처리. as_completed 는 끝난 순서대로 결과를 돌려줍니다.
    logs = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(model_path, args.native_logs)) as pool:
        # futures 를 {제출한 작업: 영상 행} 딕셔너리로 두면, 결과를 받을 때 어느 영상인지 알 수 있습니다.
        futures = {pool.submit(process_video, job): job[0] for job in jobs}
        for future in tqdm(as_completed(futures), total=len(futures), unit="영상"):
            try:
                logs.append(future.result())
            except Exception as error:
                # 자식 프로세스가 통째로 죽으면(C++ 쪽 강제 종료 등) 여기로 옵니다. 남은 작업도 함께 실패로 돌아옵니다.
                # .npz 가 저장되지 않았으니, 같은 명령을 다시 실행하면 이 영상들만 다시 처리합니다.
                row = futures[future]
                logs.append({
                    "video_id": row["video_id"], "label": row["label"], "frames": 0, "faces_found": 0,
                    "background_ratio_mean": float("nan"), "background_diff_mean": float("nan"),
                    "seconds": 0.0, "error": f"{type(error).__name__}: {error}",
                })

    # 4) 로그 저장과 요약
    logs.sort(key=lambda log: log["video_id"])
    with open(out_root / "extract_log.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(logs)

    failed = [log for log in logs if log["error"]]
    succeeded = [log for log in logs if not log["error"]]
    elapsed = time.time() - started
    print(f"완료 {len(succeeded)}개 / 실패 {len(failed)}개 | 전체 {elapsed / 60:.1f}분 (영상당 {elapsed / len(logs):.2f}초)")
    for log in failed[:5]:
        print(f"  실패: {log['video_id']} - {log['error']}")
    if succeeded:
        found_rate = sum(log["faces_found"] for log in succeeded) / sum(log["frames"] for log in succeeded)
        ratios = np.array([log["background_ratio_mean"] for log in succeeded])
        print(f"얼굴 찾은 프레임 비율 {found_rate:.3f} | 배경 면적 비율 중앙값 {np.median(ratios):.2f}, "
              f"0.3 미만인 영상 {int((ratios < 0.3).sum())}개")


# 병렬 처리(자식 프로세스)가 있는 코드는 윈도우에서 반드시 이 조건 안에서 시작해야 합니다.
# 자식 프로세스가 이 파일을 다시 읽을 때 main() 이 또 실행되는 것을 막아 줍니다.
if __name__ == "__main__":
    main()
