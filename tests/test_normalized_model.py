# ============================================================================
# 5단계 제안 모델과 그 입력 준비 테스트
# ----------------------------------------------------------------------------
# 확인하는 것
#   ClipLoader        : 썸네일에서 얼굴 자리를 가리는지, 영상마다 다른 크기를 고정 크기로 맞추는지
#   NormalizedDetector: 출력 모양, α 범위, 학습 시작 시점에 원 논문 모델과 같은 계산인지,
#                       전역 분기·게이트까지 기울기가 흐르는지
# 실행: .venv/Scripts/python.exe -m pytest tests
# ============================================================================

import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.normalized_bclstm import NormalizedDetector  # noqa: E402
from models.paper_bclstm import PaperDetector  # noqa: E402
from utils.face_clips import ClipLoader  # noqa: E402


FRAMES = 3


# 전처리 결과 흉내: 얼굴 크롭은 상수, 썸네일은 값이 다 다르고, 얼굴 상자는 썸네일 왼쪽 위에 놓습니다.
def write_clip(root, video_id, thumb_shape=(20, 30), box=(0.0, 0.0, 40.0, 40.0), scale=0.25):
    height, width = thumb_shape
    thumbs = np.full((FRAMES, height, width), 200, dtype=np.uint8)
    np.savez_compressed(
        root / f"{video_id}.npz",
        faces=np.full((FRAMES, 8, 8, 3), 100, dtype=np.uint8),
        thumbs=thumbs,
        smoothed_boxes=np.tile(np.asarray(box, dtype=np.float32), (FRAMES, 1)),
        thumb_scale=np.float32(scale),
        face_found=np.ones(FRAMES, dtype=bool),
        error=np.array(""),
    )


# 얼굴 상자(0,0)~(40,40) x 0.25 = 썸네일의 (0,0)~(10,10) 이고, 가리는 범위는 그 2배입니다.
# 그 자리는 0 이 되고 나머지는 원래 값(200)이 남아야 합니다.
def test_clip_loader_masks_face_area(tmp_path):
    write_clip(tmp_path, "a", thumb_shape=(40, 40))
    loader = ClipLoader(tmp_path, FRAMES, with_thumbs=True, thumb_size=(40, 40), exclusion_scale=2.0)

    sample = loader("a")
    thumbs = sample["thumbs"]

    assert thumbs.shape == (FRAMES, 40, 40)
    # 가린 자리(가운데가 (5,5) 인 20x20 상자)는 0, 멀리 떨어진 구석은 그대로입니다.
    assert thumbs[0, 2, 2] == 0
    assert thumbs[0, -1, -1] == 200


# 영상마다 썸네일 크기가 달라도 같은 크기로 맞춰 줍니다(배치로 묶으려면 필요합니다).
def test_clip_loader_resizes_to_fixed_size(tmp_path):
    write_clip(tmp_path, "wide", thumb_shape=(20, 30))
    write_clip(tmp_path, "tall", thumb_shape=(30, 20))
    loader = ClipLoader(tmp_path, FRAMES, with_thumbs=True, thumb_size=(16, 24))

    assert loader("wide")["thumbs"].shape == (FRAMES, 16, 24)
    assert loader("tall")["thumbs"].shape == (FRAMES, 16, 24)


# 썸네일을 달라고 하지 않으면 얼굴 크롭만 돌려줍니다(1단계 재현과 같은 동작).
def test_clip_loader_without_thumbs(tmp_path):
    write_clip(tmp_path, "a")
    sample = ClipLoader(tmp_path, FRAMES)("a")

    assert set(sample) == {"faces"}


def small_model(seed=0):
    torch.manual_seed(seed)
    return NormalizedDetector(
        global_width=8,
        gate_hidden=8,
        alpha_max=2.0,
        pretrained_file=None,
        lstm_hidden=4,
        image_size=64,
        grad_checkpointing=False,
    )


# 출력 모양과 α 범위를 봅니다. α 는 프레임마다 스칼라 하나입니다.
def test_output_shape_and_alpha_range():
    model = small_model()
    model.eval()
    clips = torch.randn(2, FRAMES, 3, 64, 64)
    thumbs = torch.randint(0, 256, (2, FRAMES, 16, 24), dtype=torch.uint8)

    with torch.no_grad():
        logits, alpha = model(clips, thumbs, return_alpha=True)

    assert logits.shape == (2, 2)
    assert alpha.shape == (2, FRAMES)
    assert torch.all(alpha >= 0) and torch.all(alpha <= 2.0)
    # 게이트 마지막 층이 0 으로 시작하므로 처음에는 alpha_max 의 절반입니다.
    assert torch.allclose(alpha, torch.full_like(alpha, 1.0))


# project 가 0 으로 시작하므로, 학습 첫 스텝의 계산은 원 논문 모델과 완전히 같아야 합니다.
def test_starts_identical_to_paper_model():
    model = small_model(seed=1)
    model.eval()
    paper = PaperDetector(pretrained_file=None, lstm_hidden=4, image_size=64, grad_checkpointing=False)
    paper.load_state_dict({key: value for key, value in model.state_dict().items()
                           if not key.startswith(("global_branch.", "gate.", "project."))})
    paper.eval()

    clips = torch.randn(2, FRAMES, 3, 64, 64)
    thumbs = torch.randint(0, 256, (2, FRAMES, 16, 24), dtype=torch.uint8)
    with torch.no_grad():
        assert torch.allclose(model(clips, thumbs), paper(clips), atol=1e-5)


# 전역 분기와 게이트까지 기울기가 흘러야 합니다(0 으로 시작해도 학습이 시작되는지).
def test_gradients_reach_global_branch_and_gate():
    model = small_model(seed=2)
    model.train()
    clips = torch.randn(2, FRAMES, 3, 64, 64)
    thumbs = torch.rand(2, FRAMES, 16, 24)

    model(clips, thumbs).sum().backward()

    assert model.project.weight.grad is not None
    assert model.project.weight.grad.abs().sum() > 0
    assert model.global_branch.net[0].weight.grad is not None
    assert model.gate.net[0].weight.grad is not None


# 배경 변화량 d_t: 첫 프레임은 0, 나머지는 프레임 사이 평균 절대 차이입니다.
def test_frame_difference():
    thumbs = torch.zeros(1, 3, 2, 2)
    thumbs[:, 1] = 0.5
    thumbs[:, 2] = 1.0

    difference = NormalizedDetector.frame_difference(thumbs)

    assert difference.shape == (1, 3, 1)
    assert difference[0, 0, 0] == 0.0
    assert torch.allclose(difference[0, 1, 0], torch.tensor(0.5))
    assert torch.allclose(difference[0, 2, 0], torch.tensor(0.5))


# 0~255 로 들어오든 0~1 로 들어오든 같은 결과가 나와야 합니다.
def test_thumb_scaling_is_consistent():
    raw = torch.randint(0, 256, (1, FRAMES, 8, 8), dtype=torch.uint8)
    scaled = raw.float() / 255.0

    assert torch.allclose(NormalizedDetector.prepare_thumbs(raw), NormalizedDetector.prepare_thumbs(scaled))
