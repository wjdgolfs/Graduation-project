# ============================================================================
# 원 논문 재현 모델과 학습 도구 테스트
# ----------------------------------------------------------------------------
# 사전학습 가중치 없이, CPU 에서 작은 입력으로 확인합니다.
# 실행: .venv/Scripts/python.exe -m pytest tests
# ============================================================================

import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.paper_bclstm import CBAM, BiConvLSTM, ConvLSTMCell, PaperDetector, prepare_clips  # noqa: E402
from utils.face_clips import PairedBatchSampler  # noqa: E402
from utils.metrics import classification_metrics  # noqa: E402


# 어텐션 가중치를 모두 0 으로 두면 Mc = Ms = σ(0) = 0.5 라서 F'' = 0.25 F, F̄ = F + F'' = 1.25 F 여야 합니다(식 2~4).
def test_cbam_follows_equations_2_to_4():
    cbam = CBAM(channels=32, reduction=16, spatial_kernel=7)
    for parameter in cbam.parameters():
        torch.nn.init.zeros_(parameter)
    features = torch.randn(2, 32, 8, 8)
    assert torch.allclose(cbam(features), 1.25 * features, atol=1e-6)


# 상태 모양은 [N, hidden, H, W] 이고, 편향은 망각 게이트(두 번째 몫)만 1 로 시작해야 합니다.
def test_convlstm_cell_shapes_and_forget_bias():
    cell = ConvLSTMCell(input_channels=6, hidden_channels=4, kernel_size=3)
    x = torch.randn(2, 6, 5, 5)
    h, c = cell(x, cell.initial_state(x))
    assert h.shape == c.shape == (2, 4, 5, 5)
    bias = cell.gates.bias.detach()
    assert torch.all(bias[4:8] == 1) and torch.all(bias[:4] == 0) and torch.all(bias[8:] == 0)


# 핍홀 가중치가 0 이 아니면 게이트가 셀 상태를 보게 되어 결과가 달라져야 합니다.
def test_convlstm_peephole_uses_cell_state():
    torch.manual_seed(0)
    cell = ConvLSTMCell(input_channels=3, hidden_channels=2, kernel_size=3, peephole=True)
    x = torch.randn(1, 3, 4, 4)
    state = (torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4))
    before = cell(x, state)[0]
    with torch.no_grad():
        cell.w_cf.fill_(1.0)
    assert not torch.allclose(before, cell(x, state)[0])


# 역방향 셀에 순방향 셀과 같은 가중치를 넣으면, 시퀀스를 뒤집어 넣었을 때 두 방향의 출력이 서로 바뀌어야 합니다.
def test_biconvlstm_backward_reads_reversed_sequence():
    torch.manual_seed(0)
    model = BiConvLSTM(input_channels=3, hidden_channels=2, kernel_size=3)
    model.backward_cell.load_state_dict(model.forward_cell.state_dict())
    sequence = torch.randn(1, 5, 3, 4, 4)
    output = model(sequence)
    flipped = model(sequence.flip(1))
    assert output.shape == (1, 4, 4, 4)
    assert torch.allclose(output[:, :2], flipped[:, 2:], atol=1e-6)
    assert torch.allclose(output[:, 2:], flipped[:, :2], atol=1e-6)


# 작은 입력으로 전체 모델의 출력 모양을 확인합니다. gradient checkpointing 을 켜도 출력이 같아야 하고
# (구간 목록이 forward_features 와 같은 계산인지), XceptionNet 첫 합성곱까지 기울기가 흘러야 합니다.
def test_detector_shapes_and_checkpointing_match():
    torch.manual_seed(0)
    model = PaperDetector(pretrained_file=None, lstm_hidden=4, image_size=64, grad_checkpointing=False)
    model.train()
    clips = torch.randn(2, 3, 3, 64, 64)
    plain = model(clips)
    model.grad_checkpointing = True
    checkpointed = model(clips)
    assert plain.shape == (2, 2)
    assert torch.allclose(plain, checkpointed, atol=1e-5)
    checkpointed.sum().backward()
    assert model.backbone.conv1.weight.grad is not None


def test_prepare_clips_scales_and_permutes():
    faces = torch.zeros(1, 2, 4, 5, 3, dtype=torch.uint8)
    faces[..., 0] = 255
    clips = prepare_clips(faces)
    assert clips.shape == (1, 2, 3, 4, 5)
    assert torch.all(clips[:, :, 0] == 1) and torch.all(clips[:, :, 1:] == -1)


# 한 에폭에 fake 는 모두 정확히 한 번, real 은 고르게 쓰이고, 배치마다 real 1개 + fake 1개여야 합니다(논문 Table 4).
def test_paired_batch_sampler_follows_table_4():
    labels = [0] * 3 + [1] * 7
    sampler = PairedBatchSampler(labels, pairs_per_batch=1, seed=0)
    batches = list(sampler)
    assert len(batches) == len(sampler) == 7
    assert all(labels[real] == 0 and labels[fake] == 1 for real, fake in batches)
    assert sorted(fake for _, fake in batches) == list(range(3, 10))
    counts = np.bincount([real for real, _ in batches], minlength=3)
    assert counts.max() - counts.min() <= 1
    # 같은 에폭이면 같은 순서, 다른 에폭이면 다른 순서
    assert batches == list(sampler)
    sampler.set_epoch(1)
    assert batches != list(sampler)


def test_classification_metrics_fake_is_positive():
    labels = [1, 1, 1, 0, 0]
    probabilities = [0.9, 0.8, 0.3, 0.6, 0.1]
    metrics = classification_metrics(labels, probabilities)
    # fake 판정(>= 0.5)은 0.9, 0.8, 0.6 → TP 2, FP 1(0.6), FN 1(0.3), TN 1(0.1)
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (2, 1, 1, 1)
    assert np.isclose(metrics["accuracy"], 3 / 5)
    assert np.isclose(metrics["precision"], 2 / 3) and np.isclose(metrics["recall"], 2 / 3)
    # fake-real 쌍 6개 중 fake 점수가 더 큰 쌍이 5개(0.3 < 0.6 만 반대)
    assert np.isclose(metrics["auc"], 5 / 6)
