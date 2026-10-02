# ============================================================================
# 전환 증강: 학습 중에 입력 안에 장면 전환을 만들어 넣습니다
# ----------------------------------------------------------------------------
# 무엇을 하나
#   학습 샘플(영상 하나의 얼굴 크롭 30프레임)에 같은 라벨의 다른 영상을 이어 붙여
#   하드컷 / 페이드 / 디졸브를 만듭니다. 정답(real / fake)은 바뀌지 않습니다.
#
#     A 영상 30프레임:  A0 A1 ... A14 | A15 ... A29
#     디졸브 8프레임  :  A0 ... A10  [A11~A18 이 A 와 B 가 섞인 프레임]  B19 ... B29
#
# 왜 필요한가
#   1) 5단계 기준선 B1 "전환 증강만 한 모델" 을 만들기 위해서입니다.
#      제안 모델이 이 기준선을 못 이기면 구조를 추가한 기여가 없습니다. 먼저 정직하게 비교할 상대를 만듭니다.
#   2) 제안 모델도 전환을 봐야 정규화 강도 α 를 배울 수 있습니다. 전환이 없는 입력만 보면 α 가 상수로 수렴합니다.
#
# 라벨이 바뀌지 않는 이유
#   real 은 real 과, fake 는 "같은 기법의" fake 와만 붙입니다(3단계 벤치마크와 같은 규칙).
#   real + real 은 두 장면 모두 진짜라서 real 이고, fake + fake 는 두 장면 모두 가짜라서 fake 입니다.
#
# 전환 길이를 24프레임 이하로 두는 이유
#   입력이 30프레임입니다. 전환이 그보다 길면 입력 전체가 전환 구간이 되어 "전환 밖" 프레임이 사라집니다.
#   벤치마크의 64프레임 디졸브는 그런 경우라서, 학습 증강에서는 8~24 만 씁니다.
#
# 알아 둘 근사 (벤치마크 클립과 다른 점)
#   벤치마크 클립은 "영상을 합성한 뒤 다시 압축하고, 그 영상에서 얼굴을 다시 찾은" 결과입니다.
#   학습 증강은 이미 잘라 둔 얼굴 크롭을 섞습니다. 그래서
#     - 블렌딩 뒤 재압축이 없고,
#     - 두 얼굴이 각자 가운데 맞춰진 상태로 겹칩니다(실제 디졸브에서는 두 얼굴 위치가 다릅니다).
#   두 번째 차이를 줄이려고 뒤 영상(B)의 크롭을 조금씩 흔듭니다(partner_jitter).
#   평가는 항상 실제로 합성한 벤치마크 클립으로 하므로, 이 근사가 결과를 부풀릴 일은 없습니다.
#
# 전환 계산은 0단계 파일럿 구현(preprocessing/synthesize_transitions.compose_transition)을 그대로 씁니다.
# 같은 계산을 두 번 구현하면 학습과 평가의 전환이 달라집니다.
# ============================================================================

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.synthesize_transitions import compose_transition, transition_window  # noqa: E402


# 프레임을 세로 dy, 가로 dx 만큼 밀고 빈 곳은 가장자리 값으로 채웁니다.
# 두 얼굴이 정확히 겹친 모양을 모델이 "전환 신호" 로 외우지 않게 뒤 영상에만 씁니다.
def shift_frames(frames, dy, dx):
    if dy == 0 and dx == 0:
        return frames
    pad = [(0, 0)] * frames.ndim
    pad[1] = (abs(dy), abs(dy))
    pad[2] = (abs(dx), abs(dx))
    padded = np.pad(frames, pad, mode="edge")
    top = abs(dy) - dy
    left = abs(dx) - dx
    return padded[:, top:top + frames.shape[1], left:left + frames.shape[2]]


# 전환 종류와 길이 목록을 (종류, 길이) 쌍으로 펼칩니다. 설정의 transitions 는 {"dissolve": [8, 16], ...} 형태입니다.
def transition_choices(settings):
    choices = []
    for kind, lengths in settings["transitions"].items():
        for length in lengths:
            choices.append((kind, int(length)))
    if not choices:
        raise ValueError("transition_augment.transitions 가 비어 있습니다.")
    return choices


class TransitionAugment:
    # rows   : 짝으로 쓸 수 있는 영상 목록(학습 분할)
    # loader : utils.face_clips.ClipLoader. 짝 영상도 본 영상과 같은 준비 과정을 거쳐야 섞을 수 있습니다
    #          (배경 가리기, 썸네일 크기 맞추기). 그래서 읽기를 직접 하지 않고 같은 loader 를 씁니다.
    def __init__(self, rows, loader, settings):
        self.loader = loader
        self.settings = settings
        self.frames = loader.frames
        self.probability = float(settings["probability"])
        self.choices = transition_choices(settings)
        self.center_jitter = int(settings["center_jitter"])
        self.partner_jitter = float(settings["partner_jitter"])
        self.seed = int(settings["seed"])

        # 짝 후보를 (기법, 라벨)별로 모읍니다. real 은 real 과, fake 는 같은 기법의 fake 와만 붙습니다.
        self.pools = {}
        for row in rows:
            key = (row.get("source", ""), int(row["label"]))
            self.pools.setdefault(key, []).append(row["video_id"])

        # 난수 생성기는 DataLoader 자식 프로세스에서 처음 쓸 때 만듭니다(프로세스마다 다른 순서가 나오도록).
        self._rng = None

    # 자식 프로세스마다 [seed, worker_id + 1] 로 갈라진 생성기를 만듭니다(0 은 자식 프로세스가 없을 때).
    # default_rng 의 씨앗 목록에는 음수를 넣을 수 없어 1 을 더합니다.
    # 같은 seed·worker 수에서는 같은 순서가 나오지만, worker 수를 바꾸면 순서가 달라집니다.
    def rng(self):
        if self._rng is None:
            worker = 0
            try:
                from torch.utils.data import get_worker_info
                info = get_worker_info()
                if info is not None:
                    worker = info.id + 1
            except ImportError:
                pass
            self._rng = np.random.default_rng([self.seed, worker])
        return self._rng

    # 같은 (기법, 라벨) 안에서 자기 자신이 아닌 영상을 하나 고릅니다. 후보가 없으면 None 입니다.
    # 자기 자신을 뺀 목록에서 뽑습니다. "뽑고 자기면 다시 뽑기" 로 하면 후보가 둘뿐일 때 실패할 수 있습니다.
    def pick_partner(self, row):
        pool = [video_id for video_id in self.pools.get((row.get("source", ""), int(row["label"])), [])
                if video_id != row["video_id"]]
        if not pool:
            return None
        return pool[self.rng().integers(len(pool))]

    # 전환 중심을 고릅니다. 전환 구간이 입력 안에 다 들어오도록 범위를 좁힌 뒤 흔듭니다.
    def pick_center(self, length):
        low = length // 2
        high = self.frames - (length - length // 2)
        if low > high:
            return None
        middle = self.frames // 2
        rng = self.rng()
        jitter = int(rng.integers(-self.center_jitter, self.center_jitter + 1)) if self.center_jitter > 0 else 0
        return int(np.clip(middle + jitter, low, high))

    # sample: {"faces": [F,H,W,3] uint8, "thumbs": [F,h,w] uint8(있을 때)}
    # 반환값: (전환을 넣은 sample, 무엇을 넣었는지 적은 info 또는 None)
    def __call__(self, sample, row):
        rng = self.rng()
        if rng.random() >= self.probability:
            return sample, None

        partner_id = self.pick_partner(row)
        if partner_id is None:
            return sample, None
        partner = self.loader(partner_id)
        if partner is None or any(key not in partner for key in sample):
            return sample, None

        kind, length = self.choices[rng.integers(len(self.choices))]
        center = self.pick_center(length)
        if center is None:
            return sample, None

        # 뒤 영상의 얼굴 크롭만 조금 흔듭니다. 썸네일은 화면 전체라서 흔들지 않습니다.
        if self.partner_jitter > 0 and "faces" in partner:
            limit = max(1, int(round(self.partner_jitter * partner["faces"].shape[1])))
            dy = int(rng.integers(-limit, limit + 1))
            dx = int(rng.integers(-limit, limit + 1))
            partner["faces"] = shift_frames(partner["faces"], dy, dx)

        mixed = {}
        for key, array in sample.items():
            # 두 배열의 모양이 다르면(해상도가 다른 영상) 그 항목은 섞지 않고 그대로 둡니다.
            if partner[key].shape != array.shape:
                mixed[key] = array
                continue
            mixed[key] = compose_transition(array, partner[key], kind, length, center)

        window = transition_window(kind, length, center)
        info = {
            "kind": kind,
            "length": length,
            "center": center,
            "window_start": window[0],
            "window_end": window[1],
            "partner": partner_id,
        }
        return mixed, info


# config.yaml 의 transition_augment 설정으로 증강기를 만듭니다. enabled 가 false 면 None 을 돌려줍니다.
def build_augment(settings, rows, loader):
    if not settings or not settings.get("enabled"):
        return None
    return TransitionAugment(rows, loader, settings)
