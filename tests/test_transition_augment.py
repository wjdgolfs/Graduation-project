# ============================================================================
# 전환 증강 테스트
# ----------------------------------------------------------------------------
# 확인하는 것
#   - 전환을 넣어도 정답(라벨)이 바뀌지 않도록 같은 기법·라벨의 영상끼리만 붙는지
#   - 얼굴 크롭과 배경 썸네일에 "같은" 전환이 들어가는지 (전역 분기가 같은 전환을 봐야 합니다)
#   - 전환 구간이 입력 30프레임 안에 다 들어오는지 (길이 24 까지)
#   - probability 0 이면 원본이 그대로 나오는지
# 실행: .venv/Scripts/python.exe -m pytest tests
# ============================================================================

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.synthesize_transitions import transition_window  # noqa: E402
from utils.face_clips import FaceClipDataset  # noqa: E402
from utils.transition_augment import TransitionAugment, build_augment, shift_frames  # noqa: E402


FRAMES = 30
SETTINGS = {
    "enabled": True,
    "probability": 1.0,
    "transitions": {"hard_cut": [0], "fade": [8, 24], "dissolve": [8, 24]},
    "center_jitter": 3,
    "partner_jitter": 0.0,
    "seed": 0,
}


# 영상마다 값이 다른 전처리 결과를 만듭니다(값만 보고 어느 영상인지 알 수 있게).
def write_clip(root, video_id, value, frames=FRAMES):
    faces = np.full((frames, 16, 16, 3), value, dtype=np.uint8)
    thumbs = np.full((frames, 8, 10), value, dtype=np.uint8)
    np.savez_compressed(
        root / f"{video_id}.npz",
        faces=faces,
        thumbs=thumbs,
        face_found=np.ones(frames, dtype=bool),
        error=np.array(""),
    )


@pytest.fixture
def clips(tmp_path):
    rows = [
        {"video_id": "real_a", "source": "original", "label": 0},
        {"video_id": "real_b", "source": "original", "label": 0},
        {"video_id": "df_a", "source": "Deepfakes", "label": 1},
        {"video_id": "df_b", "source": "Deepfakes", "label": 1},
        {"video_id": "f2f_a", "source": "Face2Face", "label": 1},
    ]
    for index, row in enumerate(rows):
        write_clip(tmp_path, row["video_id"], 10 + 40 * index)
    return rows, tmp_path


# 같은 기법·라벨 안에서만 짝을 고릅니다. 자기 자신은 고르지 않습니다.
def test_partner_has_same_source_and_label(clips):
    rows, root = clips
    augment = TransitionAugment(rows, root, SETTINGS, FRAMES)
    lookup = {row["video_id"]: row for row in rows}

    for row in rows:
        partner = augment.pick_partner(row)
        if partner is None:
            # Face2Face 는 후보가 하나뿐이라 짝이 없습니다. 그때는 증강을 건너뜁니다.
            assert row["video_id"] == "f2f_a"
            continue
        assert partner != row["video_id"]
        assert lookup[partner]["source"] == row["source"]
        assert lookup[partner]["label"] == row["label"]


# 짝이 없으면 원본을 그대로 돌려줍니다(증강 때문에 학습이 멈추면 안 됩니다).
def test_no_partner_returns_original(clips):
    rows, root = clips
    augment = TransitionAugment(rows, root, SETTINGS, FRAMES)
    row = next(row for row in rows if row["video_id"] == "f2f_a")
    with np.load(root / "f2f_a.npz") as data:
        faces = data["faces"][:FRAMES]

    sample, info = augment({"faces": faces}, row)

    assert info is None
    assert np.array_equal(sample["faces"], faces)


# 얼굴과 썸네일에 같은 전환이 들어가야 합니다. 전환 구간 밖은 양쪽 모두 원본과 같아야 합니다.
def test_faces_and_thumbs_share_the_same_window(clips):
    rows, root = clips
    augment = TransitionAugment(rows, root, SETTINGS, FRAMES)
    row = next(row for row in rows if row["video_id"] == "real_a")
    with np.load(root / "real_a.npz") as data:
        original = {"faces": data["faces"][:FRAMES], "thumbs": data["thumbs"][:FRAMES]}

    seen = set()
    for _ in range(20):
        sample, info = augment({key: value.copy() for key, value in original.items()}, row)
        assert info is not None
        seen.add(info["kind"])
        start, end = info["window_start"], info["window_end"]
        assert (start, end) == transition_window(info["kind"], info["length"], info["center"])
        assert 0 <= start <= end <= FRAMES

        for key in ("faces", "thumbs"):
            # 전환 구간 앞은 A 그대로입니다.
            assert np.array_equal(sample[key][:start], original[key][:start])
            # 전환이 들어갔으니 어딘가는 달라야 합니다.
            assert not np.array_equal(sample[key], original[key])
            # 전환 구간 뒤는 짝 영상의 값(상수)입니다. 전환이 마지막 프레임에서 끝나면 뒤가 없습니다.
            if end < FRAMES:
                assert not np.array_equal(sample[key][end:], original[key][end:])

    assert seen == {"hard_cut", "fade", "dissolve"}


# 디졸브는 알파 블렌딩입니다. 가운데 프레임은 두 영상의 평균에 가까워야 합니다.
def test_dissolve_midpoint_is_the_average(clips):
    rows, root = clips
    settings = dict(SETTINGS, transitions={"dissolve": [8]}, center_jitter=0)
    augment = TransitionAugment(rows, root, settings, FRAMES)
    row = next(row for row in rows if row["video_id"] == "real_a")
    with np.load(root / "real_a.npz") as data:
        faces = data["faces"][:FRAMES]

    sample, info = augment({"faces": faces.copy()}, row)
    with np.load(root / f"{info['partner']}.npz") as data:
        partner = data["faces"][:FRAMES]

    start, end = info["window_start"], info["window_end"]
    count = end - start
    for k in range(count):
        p = (k + 1) / (count + 1)
        expected = (1.0 - p) * float(faces[0, 0, 0, 0]) + p * float(partner[0, 0, 0, 0])
        assert abs(float(sample["faces"][start + k, 0, 0, 0]) - expected) <= 1.0


# probability 0 이면 아무것도 건드리지 않습니다.
def test_probability_zero_keeps_original(clips):
    rows, root = clips
    augment = TransitionAugment(rows, root, dict(SETTINGS, probability=0.0), FRAMES)
    row = rows[0]
    with np.load(root / "real_a.npz") as data:
        faces = data["faces"][:FRAMES]

    sample, info = augment({"faces": faces.copy()}, row)

    assert info is None
    assert np.array_equal(sample["faces"], faces)


# build_augment 는 enabled 가 꺼져 있으면 None 을 돌려줍니다(1단계 재현과 같은 조건).
def test_build_augment_respects_enabled(clips):
    rows, root = clips
    assert build_augment(dict(SETTINGS, enabled=False), rows, root, FRAMES) is None
    assert build_augment(SETTINGS, rows, root, FRAMES) is not None


# 데이터셋이 증강을 통과시키고, with_thumbs 로 썸네일까지 돌려줍니다.
def test_dataset_applies_augment_and_returns_thumbs(clips):
    rows, root = clips
    augment = TransitionAugment(rows, root, SETTINGS, FRAMES)
    dataset = FaceClipDataset(rows, root, FRAMES, augment=augment, with_thumbs=True)

    faces, thumbs, label, video_id = dataset[0]

    assert faces.shape == (FRAMES, 16, 16, 3)
    assert thumbs.shape == (FRAMES, 8, 10)
    assert label == rows[0]["label"]
    assert video_id == rows[0]["video_id"]
    # 증강이 들어갔으므로 뒤쪽 프레임은 원본(상수 10)과 달라야 합니다.
    assert faces[-1, 0, 0, 0].item() != 10

    # 증강을 주지 않으면 예전 형태(3개)를 그대로 돌려줍니다.
    plain = FaceClipDataset(rows, root, FRAMES)[0]
    assert len(plain) == 3
    assert plain[0][-1, 0, 0, 0].item() == 10


# 밀기(jitter)는 모양을 유지하고 빈 곳을 가장자리 값으로 채웁니다.
def test_shift_frames_keeps_shape_and_pads_with_edge():
    frames = np.arange(2 * 4 * 5 * 3, dtype=np.uint8).reshape(2, 4, 5, 3)

    assert np.array_equal(shift_frames(frames, 0, 0), frames)

    # 아래로 1칸: 첫 줄은 원본 첫 줄이 복제되고, 나머지는 한 줄씩 내려갑니다.
    down = shift_frames(frames, 1, 0)
    assert down.shape == frames.shape
    assert np.array_equal(down[:, 0], frames[:, 0])
    assert np.array_equal(down[:, 1:], frames[:, :-1])

    # 왼쪽으로 1칸: 마지막 열이 복제되고, 나머지는 한 칸씩 당겨집니다.
    left = shift_frames(frames, 0, -1)
    assert left.shape == frames.shape
    assert np.array_equal(left[:, :, -1], frames[:, :, -1])
    assert np.array_equal(left[:, :, :-1], frames[:, :, 1:])
