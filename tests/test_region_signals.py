# ============================================================================
# 2단계 부위 신호 계산 테스트
# ----------------------------------------------------------------------------
# 결과를 손으로 계산할 수 있는 작은 합성 예제로, 부위 마스크와 변화량이 맞는 곳에서 나오는지 확인합니다.
# 실행: .venv/Scripts/python.exe -m pytest tests
# ============================================================================

import importlib.util
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.face_regions import hull_mask, inside_frame_mask, smooth_landmarks  # noqa: E402
from preprocessing.region_signals import REGION_NAMES, compute_signals  # noqa: E402


# 가만히 있는 점은 그대로여야 하고, 일정한 속도로 움직이는 점도 양 끝을 뺀 프레임에서는 그대로여야 합니다
# (가운데 정렬 평균은 움직임에 뒤처지지 않음).
def test_smooth_landmarks_keeps_constant_and_linear_motion():
    constant = np.tile([[[10.0, 20.0]]], (5, 3, 1))
    assert np.allclose(smooth_landmarks(constant, 3), constant)

    linear = np.stack([np.full((3, 2), float(t)) for t in range(6)])
    assert np.allclose(smooth_landmarks(linear, 3)[1:-1], linear[1:-1])
    assert np.allclose(smooth_landmarks(linear, 1), linear)


# 얼굴을 못 찾은 프레임(nan)은 앞뒤 프레임 평균으로 채우고, 창 안에 쓸 프레임이 없으면 nan 으로 남아야 합니다.
def test_smooth_landmarks_fills_missing_from_neighbors():
    points = np.stack([np.full((2, 2), float(t)) for t in range(5)])
    points[2] = np.nan
    assert np.allclose(smooth_landmarks(points, 3)[2], 2.0)  # (1 + 3) / 2

    lonely = np.full((3, 2, 2), np.nan)
    lonely[0] = 1.0
    assert np.isnan(smooth_landmarks(lonely, 3)[2]).all()


# margin 을 주면 볼록 껍질이 사방으로 넓어지고, 원래 껍질은 그대로 포함해야 합니다.
def test_hull_mask_margin_expands_region():
    points = np.array([[50, 50], [70, 50], [70, 70], [50, 70]], dtype=float)
    plain = hull_mask(points, [0, 1, 2, 3], 120, 120)
    wide = hull_mask(points, [0, 1, 2, 3], 120, 120, margin=5)
    assert wide.sum() > plain.sum()
    assert wide[45, 60] == 1 and plain[45, 60] == 0
    assert np.all(wide[plain == 1] == 1)


# 상자가 프레임 왼쪽 밖으로 나가면, 크롭의 왼쪽 부분은 프레임 밖(False)이어야 합니다.
def test_inside_frame_mask_marks_padding():
    # 한 변 200 인 상자를 100 으로 줄이므로 크롭 1픽셀 = 원래 2픽셀. 원래 x < 0 인 50픽셀은 크롭의 왼쪽 25열입니다.
    inside = inside_frame_mask([-50, 0, 150, 200], frame_width=300, frame_height=300, size=100)
    assert not inside[:, :25].any()
    assert inside[:, 25:].all()


# 입 안쪽 픽셀만 바뀐 영상: 입 변화량만 커지고 눈·경계는 0, 얼굴 전체는 입이 차지하는 비율만큼만 커져야 합니다.
def test_compute_signals_localizes_change_to_mouth():
    size = 240
    points = np.array([
        [60, 60], [180, 60], [180, 180], [60, 180],  # 0~3 얼굴 윤곽
        [130, 95], [150, 95], [140, 105],            # 4~6 왼쪽 눈
        [90, 95], [110, 95], [100, 105],             # 7~9 오른쪽 눈
        [100, 140], [140, 140], [120, 160],          # 10~12 입술
    ], dtype=float)
    indices = {"face_oval": [0, 1, 2, 3], "left_eye": [4, 5, 6], "right_eye": [7, 8, 9], "lips": [10, 11, 12]}
    frames = 3
    faces = np.zeros((frames, size, size, 3), dtype=np.uint8)
    mouth = hull_mask(points, indices["lips"], size, size).astype(bool)
    faces[2][mouth] = 200
    data = {
        "faces": faces,
        "landmarks": np.tile(points, (frames, 1, 1)),
        "crop_boxes": np.tile([0.0, 0.0, 240.0, 240.0], (frames, 1)),
        "width": 240,
        "height": 240,
        "background_diff": np.array([np.nan, 0.1, 0.2], dtype=np.float32),
        "background_ratio": np.full(frames, 0.8, dtype=np.float32),
        "frame_indices": np.arange(frames),
    }
    settings = {"landmark_smoothing": 1, "eye_margin": 0, "mouth_margin": 0, "boundary_band": 3, "min_region_pixels": 1}

    result = compute_signals(data, indices, settings)
    diff = result["region_diff"]
    column = {name: index for index, name in enumerate(REGION_NAMES)}

    assert np.isnan(diff[0]).all()
    assert np.allclose(diff[1], 0)
    assert np.isclose(diff[2, column["mouth"]], 200 / 255)
    assert diff[2, column["eyes"]] == 0 and diff[2, column["boundary"]] == 0
    face_pixels = result["region_area"][2, column["face"]]
    assert np.isclose(diff[2, column["face"]], 200 / 255 * mouth.sum() / face_pixels)
    assert np.allclose(result["landmark_motion"][1:], 0)
    assert np.allclose(result["background_diff"][1:], [0.1, 0.2])


# scripts 폴더는 패키지가 아니라서 파일 경로로 직접 불러옵니다.
_summary_spec = importlib.util.spec_from_file_location(
    "summarize_regions", PROJECT_ROOT / "scripts" / "summarize_regions.py")
summarize = importlib.util.module_from_spec(_summary_spec)
_summary_spec.loader.exec_module(summarize)


# 같은 장면 쌍 찾기: fake 이름에서 그 영상을 만든 원본 real 영상 이름이 나와야 합니다.
# 이 규칙이 틀리면 "같은 장면 비교" 결과 전체가 틀어집니다.
def test_source_video_pairing():
    assert summarize.source_video("Celeb-synthesis/id0_id16_0000", "celebdf") == "Celeb-real/id0_0000"
    assert summarize.source_video("Celeb-real/id0_0000", "celebdf") is None
    assert summarize.source_video("Deepfakes/000_003", "ffpp") == "original/000"
    # 폴더 이름에 숫자가 들어가는 기법(Face2Face)도 짝을 찾아야 합니다.
    assert summarize.source_video("Face2Face/033_097", "ffpp") == "original/033"
    assert summarize.source_video("NeuralTextures/935_910", "ffpp") == "original/935"
    assert summarize.source_video("original/000", "ffpp") is None


# 얼굴 전체 대비 비율 지표 계산과, 값이 비어 있을 때의 처리
def test_metrics_of_ratios_and_missing_values():
    values = summarize.metrics_of(
        {"d_eyes": "0.02", "d_mouth": "0.03", "d_boundary": "0.015", "d_face": "0.01", "d_bg": "0.001"})
    assert np.isclose(values["boundary/face"], 1.5)
    assert np.isclose(values["eyes/face"], 2.0)
    assert np.isclose(values["mouth/face"], 3.0)

    empty = summarize.metrics_of({"d_eyes": "", "d_mouth": "", "d_boundary": "", "d_face": "0", "d_bg": ""})
    assert np.isnan(empty["d_eyes"]) and np.isnan(empty["boundary/face"])
