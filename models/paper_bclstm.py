# ============================================================================
# 원 논문 재현 모델: XceptionNet + CBAM + Bidirectional ConvLSTM
# ----------------------------------------------------------------------------
# 논문: 이대현·문종섭, "Bidirectional Convolutional LSTM을 이용한 Deepfake 탐지 방법",
#       정보보호학회논문지 30(6), 2020.12, pp.1053-1065
#
# 전체 흐름 (논문 Fig. 3, 5, 6, 7). B = 배치 크기(2), F = 프레임 수(30)
#   입력 얼굴 클립                                                     [B, F, 3, 240, 240]
#   1) Convolution Block: 모든 프레임에 같은 XceptionNet → CBAM 을 적용합니다(Keras 의 TimeDistributed).
#        프레임을 배치 방향으로 펼쳐 한 번에 계산합니다                  [B*F, 3, 240, 240]
#        XceptionNet 마지막 합성곱(Fig. 3 의 Conv)까지의 특징 F          [B*F, 2048, 8, 8]
#        CBAM: F' = Mc(F) ⊗ F,  F'' = Ms(F') ⊗ F',  F̄ = F ⊕ F''        [B*F, 2048, 8, 8]
#   2) Sequential layer: 프레임을 순서대로 읽는 ConvLSTM 과 거꾸로 읽는 ConvLSTM 의
#        마지막 은닉 상태를 채널 방향으로 이어 붙입니다                   [B, 2*hidden, 8, 8]
#   3) Fully-connected layer: 펼쳐서 출력 노드 2개(real, fake)          [B, 2]
#
# 논문에 값이 없어서 정한 것 (configs/config.yaml 의 paper_model 에서 바꿀 수 있습니다)
#   - XceptionNet 초기 가중치: Keras 의 ImageNet 사전학습 가중치를 옮긴 timm "legacy_xception"
#   - CBAM 채널 축소율 r = 16 (CBAM 원 논문 기본값). 공간 어텐션의 7x7 필터는 논문 값입니다.
#   - ConvLSTM 은닉 채널 128, 필터 3x3. 망각 게이트 편향은 Keras 기본값처럼 1 로 시작합니다.
#   - 식 (7)~(11) 의 핍홀 항(W_c ⊗ c)은 채널마다 가중치 하나를 모든 위치에 함께 씁니다(peephole: false 로 끌 수 있음).
#   - 출력: 논문에 softmax(3.1절)와 sigmoid(3.4절)가 함께 적혀 있습니다. 여기서는 로짓 2개를 내고,
#     학습 코드에서 softmax + 교차 엔트로피(식 13)로 학습합니다. 판정은 둘 중 큰 쪽입니다.
# ============================================================================

from pathlib import Path

import timm
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


# 전처리 결과의 uint8 얼굴 크롭 [B, F, H, W, 3] 을 모델 입력 [B, F, 3, H, W] 실수로 바꿉니다.
# XceptionNet 사전학습 때와 같이 0~255 를 -1~1 로 맞춥니다(timm legacy_xception 의 mean = std = 0.5).
# GPU 로 옮긴 뒤에 부르면 이 계산도 GPU 에서 합니다.
def prepare_clips(faces):
    return faces.permute(0, 1, 4, 2, 3).float() / 127.5 - 1.0


# CBAM 1단계, 채널 어텐션 Mc (식 5): 특징 지도에서 "무엇(어떤 채널)"이 중요한지 채널마다 0~1 가중치를 만듭니다.
#   Mc(F) = σ( MLP(AvgPool(F)) + MLP(MaxPool(F)) ),  MLP(v) = W1(ReLU(W0 v))
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # W0: C → C/r, W1: C/r → C. 1x1 합성곱은 채널 벡터에 곱하는 완전연결층과 같습니다. 식 (5)에 편향이 없어 뺍니다.
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
        )

    def forward(self, features):
        # 높이·폭 방향으로 평균/최대를 내면 [N, C, 1, 1] 이 됩니다. 두 결과는 같은 MLP 를 거칩니다.
        average = self.mlp(features.mean(dim=(2, 3), keepdim=True))
        maximum = self.mlp(features.amax(dim=(2, 3), keepdim=True))
        return torch.sigmoid(average + maximum)


# CBAM 2단계, 공간 어텐션 Ms (식 6): "어디(어느 위치)"가 중요한지 위치마다 0~1 가중치를 만듭니다.
#   Ms(F') = σ( f7x7([AvgPool(F'); MaxPool(F')]) ). 여기서 풀링은 채널 방향이라 [N, 1, H, W] 지도 2장이 나옵니다.
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        # padding = kernel_size // 2 로 두면 출력 지도의 높이·폭이 입력과 같습니다.
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, features):
        pooled = torch.cat([features.mean(dim=1, keepdim=True), features.amax(dim=1, keepdim=True)], dim=1)
        return torch.sigmoid(self.conv(pooled))


# CBAM 전체 (식 2~4). 채널 어텐션과 공간 어텐션을 차례로 곱한 뒤, 원래 특징 F 를 더합니다(잔차 연결).
# ⊗ 는 같은 위치끼리 곱하기(아다마르 곱)입니다. 모양이 다르면 크기가 1 인 축을 늘려서 곱합니다(브로드캐스팅).
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, features):
        refined = self.channel_attention(features) * features   # F'  = Mc(F) ⊗ F    (식 2)
        refined = self.spatial_attention(refined) * refined     # F'' = Ms(F') ⊗ F'  (식 3)
        return features + refined                               # F̄  = F ⊕ F''      (식 4)


# ConvLSTM 셀 하나 (식 7~11). LSTM 의 행렬 곱을 합성곱(*)으로 바꿔서, 입력과 상태가 [채널, 높이, 폭] 지도 모양 그대로 흐릅니다.
#   i_t = σ(W_xi * x_t + W_hi * h_{t-1} + W_ci ⊗ c_{t-1} + b_i)            입력 게이트: 새 정보를 얼마나 넣을지
#   f_t = σ(W_xf * x_t + W_hf * h_{t-1} + W_cf ⊗ c_{t-1} + b_f)            망각 게이트: 이전 기억을 얼마나 남길지
#   c_t = f_t ⊗ c_{t-1} + i_t ⊗ tanh(W_xc * x_t + W_hc * h_{t-1} + b_c)    셀 상태(기억)
#   o_t = σ(W_xo * x_t + W_ho * h_{t-1} + W_co ⊗ c_t + b_o)                출력 게이트: 기억을 얼마나 내보낼지
#   h_t = o_t ⊗ tanh(c_t)                                                 은닉 상태(이 시점의 출력)
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, peephole=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.peephole = peephole
        # x_t 와 h_{t-1} 을 채널 방향으로 붙여 합성곱 한 번으로 네 식의 "W_x * x_t + W_h * h_{t-1} + b" 를 함께 계산합니다.
        # 출력 채널 4*hidden 을 차례로 i, f, o, g(tanh 에 들어갈 값) 네 몫으로 나눠 씁니다.
        self.gates = nn.Conv2d(input_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=kernel_size // 2)
        nn.init.zeros_(self.gates.bias)
        # 망각 게이트 편향을 1 로 시작하면 학습 초반에 σ(1) ≈ 0.73 만큼 이전 기억을 남겨서, 앞 프레임 정보가 잘 전달됩니다.
        nn.init.ones_(self.gates.bias[hidden_channels:2 * hidden_channels])
        if peephole:
            # W_ci, W_cf, W_co: 게이트가 셀 상태 c 를 직접 보게 하는 가중치. 채널마다 값 하나([1, hidden, 1, 1])를 모든 위치에 곱합니다.
            # 0 으로 시작하므로 처음에는 핍홀이 없는 ConvLSTM 과 같고, 학습하면서 필요한 만큼 커집니다.
            self.w_ci = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
            self.w_cf = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
            self.w_co = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))

    # 첫 프레임 전의 상태 (h_0, c_0) 는 0 입니다. 높이·폭은 입력 지도와 같고 채널은 hidden 입니다.
    def initial_state(self, x):
        zeros = x.new_zeros(x.shape[0], self.hidden_channels, x.shape[2], x.shape[3])
        return zeros, zeros

    def forward(self, x, state):
        h_prev, c_prev = state
        # chunk(4, dim=1): 채널 방향으로 4등분합니다.
        i, f, o, g = self.gates(torch.cat([x, h_prev], dim=1)).chunk(4, dim=1)
        if self.peephole:
            i = i + self.w_ci * c_prev
            f = f + self.w_cf * c_prev
        i = torch.sigmoid(i)                      # (식 7)
        f = torch.sigmoid(f)                      # (식 8)
        c = f * c_prev + i * torch.tanh(g)        # (식 9)
        if self.peephole:
            o = o + self.w_co * c                 # 출력 게이트는 새 셀 상태 c_t 를 봅니다.
        o = torch.sigmoid(o)                      # (식 10)
        h = o * torch.tanh(c)                     # (식 11)
        return h, c


# 양방향 ConvLSTM (논문 Fig. 5). 순방향 셀은 1 → F 순서로, 역방향 셀은 F → 1 순서로 프레임 특징을 읽습니다.
# 두 셀은 가중치를 따로 가지며, 각자 끝까지 읽은 뒤의 은닉 상태(순방향 h_F, 역방향 h_1)를 채널 방향으로 이어 붙여 내보냅니다.
class BiConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, peephole=True):
        super().__init__()
        self.forward_cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size, peephole)
        self.backward_cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size, peephole)

    # 셀 하나로 시퀀스를 order 순서대로 읽고 마지막 은닉 상태를 돌려줍니다.
    @staticmethod
    def read(cell, sequence, order):
        state = cell.initial_state(sequence[:, 0])
        for t in order:
            state = cell(sequence[:, t], state)
        return state[0]

    def forward(self, sequence):
        # sequence: [B, F, C, H, W]
        frames = sequence.shape[1]
        forward_last = self.read(self.forward_cell, sequence, range(frames))
        backward_last = self.read(self.backward_cell, sequence, range(frames - 1, -1, -1))
        return torch.cat([forward_last, backward_last], dim=1)


# 전체 모델. clips [B, F, 3, H, W] (prepare_clips 결과) → 로짓 [B, 2] (0번 = real, 1번 = fake)
class PaperDetector(nn.Module):
    def __init__(
        self,
        backbone="legacy_xception",
        pretrained_file=None,
        cbam_reduction=16,
        spatial_kernel=7,
        lstm_hidden=128,
        lstm_kernel=3,
        peephole=True,
        image_size=240,
        num_classes=2,
        grad_checkpointing=True,
    ):
        super().__init__()
        if pretrained_file is not None and not Path(pretrained_file).exists():
            raise FileNotFoundError(
                f"XceptionNet 가중치 파일이 없습니다: {pretrained_file}\n먼저 python scripts/setup_xception.py 를 실행하세요."
            )
        # num_classes=0, global_pool="": ImageNet 분류층과 전역 풀링을 빼고 특징 지도까지만 씁니다.
        # pretrained_cfg_overlay 의 file: 인터넷에서 받지 않고 이 파일에서 사전학습 가중치를 읽게 합니다.
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained_file is not None,
            pretrained_cfg_overlay=None if pretrained_file is None else dict(file=str(pretrained_file)),
            num_classes=0,
            global_pool="",
        )
        channels = self.backbone.num_features
        self.cbam = CBAM(channels, cbam_reduction, spatial_kernel)
        self.sequence = BiConvLSTM(channels, lstm_hidden, lstm_kernel, peephole)

        # 입력 크기에 따른 특징 지도 크기(240 → 8x8)를 빈 이미지로 한 번 계산해, 완전연결층의 입력 수를 정합니다.
        # 평가 모드로 계산해야 BatchNorm 의 누적 평균·분산이 이 빈 이미지 때문에 바뀌지 않습니다.
        self.backbone.eval()
        with torch.no_grad():
            feature_size = self.backbone.forward_features(torch.zeros(1, 3, image_size, image_size)).shape[-1]
        self.backbone.train()
        self.classifier = nn.Linear(2 * lstm_hidden * feature_size * feature_size, num_classes)

        # gradient checkpointing: 학습 때 XceptionNet 의 중간 결과를 구간 경계에서만 저장하고, 역전파 때 구간 안을 다시 계산합니다.
        # 프레임 60장(배치 2 x 30)을 한 번에 넣어도 GPU 메모리에 들어가게 하는 대신 계산이 조금 늘어납니다.
        self.grad_checkpointing = grad_checkpointing
        self._stages = self.xception_stages() if backbone == "legacy_xception" else None
        if grad_checkpointing and self._stages is None:
            raise NotImplementedError("gradient checkpointing 구간은 legacy_xception 구조에 맞춰 나눴습니다.")

    # forward_features 와 같은 순서로 XceptionNet 을 14 구간으로 나눈 목록입니다.
    # 모듈 속성이 아닌 파이썬 리스트에 두어서, 같은 가중치가 state_dict 에 두 번 들어가지 않습니다.
    def xception_stages(self):
        net = self.backbone
        return [
            nn.Sequential(net.conv1, net.bn1, net.act1, net.conv2, net.bn2, net.act2),
            *[getattr(net, f"block{k}") for k in range(1, 13)],
            nn.Sequential(net.conv3, net.bn3, net.act3, net.conv4, net.bn4, net.act4),
        ]

    # 프레임 이미지 [N, 3, H, W] → XceptionNet 마지막 합성곱 특징 [N, 2048, h, w]
    def extract(self, images):
        if self.grad_checkpointing and self.training and torch.is_grad_enabled():
            # use_reentrant=False: PyTorch 가 권장하는 방식. 구간 입력이 기울기를 요구하지 않아도 가중치 기울기가 제대로 흐릅니다.
            # BatchNorm 이 든 구간을 다시 계산하면 누적 평균·분산이 한 스텝에 두 번 갱신됩니다. 흔히 무시하는 작은 차이입니다.
            for stage in self._stages:
                images = checkpoint(stage, images, use_reentrant=False)
            return images
        return self.backbone.forward_features(images)

    def forward(self, clips):
        batch, frames = clips.shape[:2]
        features = self.extract(clips.flatten(0, 1))           # [B*F, 2048, 8, 8]
        features = self.cbam(features)                          # [B*F, 2048, 8, 8]
        features = features.unflatten(0, (batch, frames))       # [B, F, 2048, 8, 8]
        states = self.sequence(features)                        # [B, 2*hidden, 8, 8]
        return self.classifier(states.flatten(1))               # [B, 2]


# config.yaml 의 paper_model 설정으로 모델을 만듭니다. pretrained=False 면 XceptionNet 을 무작위 가중치로 시작합니다(시험용).
def build_model(settings, pretrained=True):
    return PaperDetector(
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
