# ============================================================================
# 5단계 제안 모델: 배경 기준 정규화를 넣은 Bi-ConvLSTM 탐지기
# ----------------------------------------------------------------------------
# 한 줄 요약
#   원 논문 모델(models/paper_bclstm.PaperDetector)에 "전역 변화(배경) 분기" 를 붙이고,
#   얼굴 특징에서 전역 성분을 얼마나 뺄지(α_t)를 학습으로 정합니다.
#
#   신호 수준 가설:  R    = D_region − α · D_bg        (3단계에서 α = 1.27 로 측정)
#   모델 수준 구현:  F̃_t = F_t      − α_t · g_t       (α_t 는 프레임마다 정해지는 스칼라)
#
# 구조
#   얼굴 크롭 [B,F,3,240,240] ─ Xception → CBAM ─ F_t [B,F,2048,8,8] ─┐
#                                                                      │ F̃_t = F_t − α_t·g_t
#   배경 썸네일 [B,F,96,128] ─ 작은 CNN 4층 ─ b_t [B,F,64] ─ project ─ g_t [B,F,2048]
#                                 └─ gate([b_t, b_t−b_{t−1}, d_t]) ─ α_t [B,F,1]
#                                                                      ▼
#                                           Bi-ConvLSTM(은닉 128) → [B,256,8,8] → FC → real/fake
#
# 왜 이렇게 했나 (측정 근거는 docs/04_5단계_설계.md 1절)
#   - α 를 상수로 두지 않는 이유: 고정 α = 1.27 은 디졸브는 잘 지우지만 페이드를 과보정합니다(3단계).
#   - α 를 "D_bg 값의 함수" 로만 두지 않는 이유: 같은 디졸브 길이 안에서 클립별 D_bg 크기는
#     탐지기가 속는 정도를 예측하지 못했습니다(ρ ≈ 0). 그래서 게이트는 전역 특징의 시간 패턴도 함께 봅니다.
#   - α_t 가 프레임마다 스칼라 하나인 이유: 학습이 끝난 뒤 시간축에 그려서 전환 구간과 겹쳐 볼 수 있습니다.
#     "모델이 전환 구간에서 더 많이 뺀다" 를 그림 한 장으로 보일 수 있고, 고정 α 와도 비교됩니다.
#   - 정규화를 Bi-ConvLSTM "입력" 단계에서 하는 이유: 가설이 말하는 불연속은 시간 차이입니다.
#     시간 모듈에 들어가기 전에 프레임마다 전역 성분을 빼야, 시간 모듈이 전역 성분이 빠진 변화만 봅니다.
#
# 학습 시작점이 원 논문 모델과 같은 이유 (중요)
#   project 의 가중치를 0 으로 시작하므로 g_t = 0 이고, 따라서 F̃_t = F_t 입니다.
#   즉 학습 첫 스텝에서는 원 논문 모델과 완전히 같은 계산입니다. 거기서부터 정규화를 배워 갑니다.
#   α 의 시작값은 sigmoid(0) x alpha_max = 1.0 으로, 3단계에서 잰 1.27 과 가까운 곳에서 출발합니다.
#
# 입력 (utils/face_clips.ClipLoader 가 준비)
#   clips  : prepare_clips 를 거친 얼굴 크롭 [B, F, 3, H, W] (−1 ~ 1)
#   thumbs : 얼굴 자리를 가린 배경 썸네일 [B, F, h, w] uint8 또는 실수. 0~255 이면 안에서 0~1 로 바꿉니다.
# ============================================================================

import sys
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.paper_bclstm import PaperDetector  # noqa: E402


# 배경 썸네일 한 장 → 전역 특징 벡터. 합성곱 4층으로 크기를 1/16 로 줄이고 평균 풀링으로 벡터를 만듭니다.
# 얼굴 분기(Xception)보다 훨씬 작습니다. 배경의 "무엇이 찍혔나" 가 아니라 "얼마나 바뀌나" 를 보는 분기이기 때문입니다.
class GlobalBranch(nn.Module):
    def __init__(self, width=64, channels=(16, 32, 64)):
        super().__init__()
        layers = []
        in_channels = 1
        for out_channels in channels:
            layers += [
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels
        layers += [
            nn.Conv2d(in_channels, width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        ]
        self.net = nn.Sequential(*layers)
        self.width = width

    # frames: [N, 1, h, w] → [N, width]
    def forward(self, frames):
        return self.net(frames).flatten(1)


# 정규화 강도 α_t 를 정하는 게이트입니다.
#   입력: 전역 특징 b_t, 그 시간 차분 b_t − b_{t−1}, 배경 변화량 d_t (프레임 사이 평균 절대 차이)
#   출력: α_t = alpha_max x sigmoid(·) ∈ [0, alpha_max]
# sigmoid 로 묶는 이유: 음수(전역 성분을 더하는 방향)와 과도한 보정을 막습니다.
class NormalizationGate(nn.Module):
    def __init__(self, width=64, hidden=64, alpha_max=2.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * width + 1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.alpha_max = alpha_max
        # 마지막 층을 0 으로 시작하면 α 가 모든 프레임에서 alpha_max / 2 (기본값 1.0) 로 출발합니다.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    # features: [B, F, 2*width + 1] → [B, F, 1]
    def forward(self, features):
        return self.alpha_max * torch.sigmoid(self.net(features))


class NormalizedDetector(PaperDetector):
    def __init__(self, global_width=64, gate_hidden=64, alpha_max=2.0, **kwargs):
        super().__init__(**kwargs)
        channels = self.backbone.num_features
        self.global_branch = GlobalBranch(global_width)
        self.gate = NormalizationGate(global_width, gate_hidden, alpha_max)
        # b_t 를 얼굴 특징의 채널 수로 보냅니다. 공간 방향으로는 같은 값을 펼칩니다
        # (전역 성분은 화면 전체에 고르게 들어온다고 보는 가정입니다).
        self.project = nn.Linear(global_width, channels)
        # 0 으로 시작 → 첫 스텝은 원 논문 모델과 같은 계산.
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    # thumbs [B, F, h, w] (0~255 또는 0~1) → 0~1 실수로 맞춥니다.
    @staticmethod
    def prepare_thumbs(thumbs):
        thumbs = thumbs.float()
        return thumbs / 255.0 if thumbs.max() > 1.5 else thumbs

    # 배경 변화량 d_t: 프레임 사이 평균 절대 차이. 첫 프레임은 0 입니다.
    # 얼굴 자리는 두 프레임 모두 0 이라 차이에 기여하지 않습니다(ClipLoader 가 미리 가립니다).
    @staticmethod
    def frame_difference(thumbs):
        difference = (thumbs[:, 1:] - thumbs[:, :-1]).abs().flatten(2).mean(dim=2)
        return torch.cat([torch.zeros_like(difference[:, :1]), difference], dim=1).unsqueeze(-1)

    # 전역 분기와 게이트를 돌려 (전역 성분 g_t [B,F,C], 정규화 강도 α_t [B,F,1]) 를 만듭니다.
    def global_terms(self, thumbs):
        thumbs = self.prepare_thumbs(thumbs)
        batch, frames = thumbs.shape[:2]
        embedded = self.global_branch(thumbs.flatten(0, 1).unsqueeze(1)).unflatten(0, (batch, frames))
        # 시간 차분. 첫 프레임은 앞이 없으므로 0 입니다.
        change = torch.cat([torch.zeros_like(embedded[:, :1]), embedded[:, 1:] - embedded[:, :-1]], dim=1)
        alpha = self.gate(torch.cat([embedded, change, self.frame_difference(thumbs)], dim=2))
        return self.project(embedded), alpha

    # clips [B, F, 3, H, W], thumbs [B, F, h, w] → 로짓 [B, 2]
    # return_alpha=True 면 (로짓, α_t [B, F]) 를 함께 돌려줍니다(학습 뒤 해석용).
    def forward(self, clips, thumbs, return_alpha=False):
        batch, frames = clips.shape[:2]
        features = self.extract(clips.flatten(0, 1))            # [B*F, 2048, 8, 8]
        features = self.cbam(features)
        features = features.unflatten(0, (batch, frames))        # [B, F, 2048, 8, 8]

        global_term, alpha = self.global_terms(thumbs)           # [B, F, 2048], [B, F, 1]
        # 공간 방향으로 펼쳐서 뺍니다: [B, F, 2048, 1, 1] 이 [B, F, 2048, 8, 8] 에 자동으로 맞춰집니다.
        features = features - (alpha * global_term).unsqueeze(-1).unsqueeze(-1)

        states = self.sequence(features)                         # [B, 2*hidden, 8, 8]
        logits = self.classifier(states.flatten(1))
        return (logits, alpha.squeeze(-1)) if return_alpha else logits


# config.yaml 의 paper_model + normalized_model 설정으로 제안 모델을 만듭니다.
# pretrained=False 면 XceptionNet 을 무작위 가중치로 시작합니다(시험용).
def build_normalized_model(settings, extra, pretrained=True):
    return NormalizedDetector(
        global_width=extra["global_width"],
        gate_hidden=extra["gate_hidden"],
        alpha_max=extra["alpha_max"],
        backbone=settings["backbone"],
        pretrained_file=settings["backbone_weights"] if pretrained else None,
        cbam_reduction=settings["cbam_reduction"],
        spatial_kernel=settings["spatial_kernel"],
        lstm_hidden=settings["lstm_hidden"],
        lstm_kernel=settings["lstm_kernel"],
        peephole=settings["peephole"],
        image_size=settings["image_size"],
        grad_checkpointing=settings["grad_checkpointing"],
    )
