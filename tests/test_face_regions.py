# ============================================================================
# 얼굴 영역 계산과 데이터셋 목록 테스트
# ----------------------------------------------------------------------------
# 전처리 결과는 모든 학습의 입력이라, 좌표 계산이 1픽셀만 어긋나도 부위 마스크가 엉뚱한 곳을 가리킵니다.
# 결과를 계산으로 미리 알 수 있는 작은 예제로 확인합니다.
#
# 실행: .venv/Scripts/python.exe -m pytest tests
# ============================================================================

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.build_index import celebdf_index  # noqa: E402
from preprocessing.face_regions import (  # noqa: E402
    background_mask,
    boundary_band_mask,
    box_iou,
    clip_window,
    crop_frame,
    hull_mask,
    masked_mean_abs_diff,
    search_windows,
    smooth_boxes,
    square_box,
    tight_box,
    to_crop_coords,
)
from utils.video_io import read_selected_frames, write_frames  # noqa: E402


# nan 점은 무시하고, 점이 하나도 없으면 nan 상자여야 합니다.
def test_tight_box_ignores_nan_points():
    points = np.array([[10, 20], [np.nan, np.nan], [30, 5]])
    assert np.allclose(tight_box(points), [10, 5, 30, 20])
    assert np.isnan(tight_box(np.full((3, 2), np.nan))).all()


# 정사각형이 되고, 한 변은 긴 변 x scale, 중심은 그대로여야 합니다.
def test_square_box_keeps_center_and_scales_longest_side():
    box = square_box([0, 0, 100, 120], 1.5)
    assert np.allclose(box[2] - box[0], 180)
    assert np.allclose(box[3] - box[1], 180)
    assert np.allclose([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], [50, 60])


def test_box_iou():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert np.isclose(box_iou([0, 0, 10, 10], [5, 0, 15, 10]), 50 / 150)


# 흔들리지 않는 상자는 부드럽게 해도 그대로여야 하고, 못 찾은 프레임은 가까운 프레임 상자로 채워져야 합니다.
def test_smooth_boxes_fills_missing_and_keeps_constant_boxes():
    boxes = np.array([[10, 10, 50, 50]] * 5, dtype=float)
    boxes[2] = np.nan
    smoothed, found = smooth_boxes(boxes, window=3)
    assert found.tolist() == [True, True, False, True, True]
    assert np.allclose(smoothed, [[10, 10, 50, 50]] * 5)


# 좌우로 흔들리는 상자는 이동 평균 뒤에 흔들림(표준편차)이 줄어야 합니다.
def test_smooth_boxes_reduces_jitter():
    offsets = np.array([0, 4, -4, 4, -4, 4, -4, 0], dtype=float)
    boxes = np.stack([[10 + d, 10, 50 + d, 50] for d in offsets])
    smoothed, _ = smooth_boxes(boxes, window=5)
    assert smoothed[:, 0].std() < boxes[:, 0].std()


def test_smooth_boxes_with_no_face_returns_nan():
    smoothed, found = smooth_boxes(np.full((4, 4), np.nan), window=5)
    assert not found.any()
    assert np.isnan(smoothed).all()


# 크롭 이미지와 크롭 좌표 랜드마크가 같은 위치를 가리켜야 합니다.
# 흰 점 하나를 찍고 크롭한 뒤, 가장 밝은 픽셀 위치와 to_crop_coords 결과를 비교합니다.
def test_crop_frame_and_crop_coords_agree():
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    frame[40, 50] = 255
    box = [30, 20, 90, 80]  # 한 변 60
    crop = crop_frame(frame, box, 120)  # 2배로 키움

    expected = to_crop_coords([[50, 40]], box, 120)[0]
    y, x = np.unravel_index(crop[..., 0].argmax(), crop.shape[:2])
    assert np.allclose(expected, [40, 40])
    assert abs(x - expected[0]) <= 2 and abs(y - expected[1]) <= 2


# 상자가 프레임 밖으로 나가면 나간 부분은 검정으로 채워지고 오류가 나지 않아야 합니다.
def test_crop_frame_pads_outside_with_black():
    frame = np.full((50, 50, 3), 200, dtype=np.uint8)
    crop = crop_frame(frame, [-25, -25, 25, 25], 50)
    assert crop.shape == (50, 50, 3)
    assert crop[:20, :20].max() == 0
    assert crop[30:, 30:].min() > 150


# 정사각형 네 꼭짓점의 볼록 껍질은 그 정사각형을 채워야 하고, 경계 띠는 가운데가 비어 있어야 합니다.
def test_hull_and_boundary_band():
    points = np.array([[20, 20], [79, 20], [79, 79], [20, 79]], dtype=float)
    face = hull_mask(points, [0, 1, 2, 3], 100, 100)
    assert face[20:80, 20:80].all()
    assert face.sum() == 60 * 60

    band = boundary_band_mask(points, [0, 1, 2, 3], 100, 100, band=3)
    assert band[50, 50] == 0
    assert band[20, 50] == 1
    assert band[5, 5] == 0


# 배경 마스크는 넓힌 얼굴 상자 안을 빼고, 바깥만 True 여야 합니다.
def test_background_mask_excludes_expanded_face_box():
    mask = background_mask(100, 100, [40, 40, 60, 60], exclusion_scale=2.0)
    assert not mask[30:70, 30:70].any()
    assert mask[:30].all() and mask[70:].all()
    assert np.isclose(mask.mean(), 1 - 0.16)
    assert background_mask(10, 10, [np.nan] * 4, 2.0).all()


def test_masked_mean_abs_diff():
    previous = np.zeros((4, 4), dtype=np.uint8)
    current = np.full((4, 4), 51, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=bool)
    assert np.isnan(masked_mean_abs_diff(previous, current, mask))
    mask[0] = True
    assert np.isclose(masked_mean_abs_diff(previous, current, mask), 51 / 255)


# 프레임 밖으로 나간 상자는 프레임 안으로 잘려야 합니다.
def test_clip_window_stays_inside_frame():
    assert clip_window([-10.5, 5.2, 120.3, 90.9], 80, 100) == (0, 5, 100, 80)


# 얼굴 찾기 창은 모두 같은 크기의 정사각형이고, 프레임의 모든 픽셀을 덮어야 하며,
# 가장 먼저 시도하는 창은 화면 가운데를 포함해야 합니다.
def test_search_windows_cover_frame_and_start_near_center():
    height, width = 500, 944
    windows = search_windows(height, width, window_fraction=0.5, overlap=0.5)

    assert all(x2 - x1 == 250 and y2 - y1 == 250 for x1, y1, x2, y2 in windows)
    covered = np.zeros((height, width), dtype=bool)
    for x1, y1, x2, y2 in windows:
        covered[y1:y2, x1:x2] = True
    assert covered.all()

    x1, y1, x2, y2 = windows[0]
    assert x1 <= width / 2 <= x2 and y1 <= height / 2 <= y2


# 원하는 번호의 프레임만 정확히 골라 읽어야 합니다(프레임마다 밝기를 다르게 해서 확인).
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg 필요")
def test_read_selected_frames_picks_exact_indices(tmp_path):
    frames = np.stack([np.full((16, 16, 3), 10 * k, dtype=np.uint8) for k in range(20)])
    path = tmp_path / "numbers.mp4"
    write_frames(path, frames, fps=30.0, crf=10)

    selected, fps, indices = read_selected_frames(path, [15, 3, 3, 99])
    assert indices.tolist() == [3, 15]
    assert np.allclose(selected.reshape(2, -1).mean(axis=1), [30, 150], atol=4)


# Celeb-DF 목록: 논문 방식 테스트는 공식 목록 안에서 폴더마다 정해진 수만큼 뽑히고,
# YouTube-real 은 excluded, 공식 분할은 목록 파일을 그대로 따라야 합니다.
def test_celebdf_index_splits(tmp_path):
    for source, count in (("Celeb-real", 6), ("Celeb-synthesis", 8), ("YouTube-real", 3)):
        (tmp_path / source).mkdir()
        for i in range(count):
            (tmp_path / source / f"v{i}.mp4").write_bytes(b"")
    listed = ["1 Celeb-real/v0.mp4", "1 Celeb-real/v1.mp4", "1 Celeb-real/v2.mp4",
              "0 Celeb-synthesis/v0.mp4", "0 Celeb-synthesis/v1.mp4", "1 YouTube-real/v0.mp4"]
    (tmp_path / "List_of_testing_videos.txt").write_text("\n".join(listed), encoding="utf-8")

    rows = celebdf_index(tmp_path, test_per_class=2, seed=0)
    by_id = {row["video_id"]: row for row in rows}

    test_rows = [row for row in rows if row["split"] == "test"]
    assert sum(row["source"] == "Celeb-real" for row in test_rows) == 2
    assert sum(row["source"] == "Celeb-synthesis" for row in test_rows) == 2
    assert all(row["split_official"] == "test" for row in test_rows)
    assert all(row["split"] == "excluded" for row in rows if row["source"] == "YouTube-real")
    assert by_id["YouTube-real/v0"]["split_official"] == "test"
    assert by_id["Celeb-synthesis/v5"]["label"] == 1 and by_id["Celeb-real/v5"]["label"] == 0
    # 같은 seed 면 같은 영상이 뽑혀야 합니다.
    again = celebdf_index(tmp_path, test_per_class=2, seed=0)
    assert [r["video_id"] for r in again if r["split"] == "test"] == [r["video_id"] for r in test_rows]
