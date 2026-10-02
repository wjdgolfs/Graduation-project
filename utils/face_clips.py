# ============================================================================
# 전처리된 얼굴 클립(.npz)을 학습에 넣는 도구
# ----------------------------------------------------------------------------
#   load_index         : index.csv 에서 원하는 분할(train / test 등)의 영상 행만 읽습니다.
#   usable_rows        : 전처리 결과가 아직 없거나 실패한 영상을 뺍니다.
#   FaceClipDataset    : 영상 하나 → (얼굴 크롭 [F, H, W, 3] uint8 텐서, 라벨, video_id)
#                        with_thumbs=True 면 배경 썸네일([F, h, w])도 함께 돌려줍니다(5단계 전역 분기 입력).
#                        augment 를 주면 학습 중에 장면 전환을 넣습니다(utils/transition_augment.py).
#   PairedBatchSampler : 원 논문 Table 4 처럼 real 1개 + fake 1개를 한 배치로 묶습니다.
#
# 알아두면 좋은 개념
#   - Dataset: "i 번째 샘플을 달라"에 답하는 객체. DataLoader 가 여러 프로세스(num_workers)로 나눠 불러 줍니다.
#   - batch_sampler: 배치마다 어떤 샘플 번호들을 묶을지 정하는 객체. 여기서는 real / fake 수를 맞추는 데 씁니다.
#   - 이미지는 uint8 그대로 넘기고, 실수 변환은 GPU 에서 합니다(models/paper_bclstm.py 의 prepare_clips).
#     float32 는 uint8 보다 4배 커서, CPU 에서 바꾸면 프로세스 사이로 옮기는 데이터도 4배가 됩니다.
# ============================================================================

import csv
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


# index.csv(preprocessing/build_index.py 가 만든 목록)에서 split_column 값이 split 인 행만 읽습니다. 라벨은 정수로 바꿉니다.
def load_index(index_path, split, split_column="split"):
    with open(index_path, "r", newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row[split_column] == split]
    for row in rows:
        row["label"] = int(row["label"])
    return rows


# 전처리 결과(.npz)가 있고 오류 없이 끝난 영상만 남깁니다. 반환값: (남은 행 목록, 결과 없음 수, 실패 수)
# 저장 중 PC 가 꺼져 열리지 않는 파일도 "결과 없음"으로 셉니다. 전처리를 다시 실행하면 새로 만들어집니다.
def usable_rows(rows, npz_root):
    kept, missing, failed = [], 0, 0
    for row in rows:
        path = Path(npz_root) / f"{row['video_id']}.npz"
        try:
            # np.load 는 .npz 안의 배열을 꺼낼 때 그 배열만 풀기 때문에, 작은 error 값만 읽으면 빠릅니다.
            with np.load(path) as data:
                error = str(data["error"]).strip()
        except (OSError, KeyError, zipfile.BadZipFile):
            missing += 1
            continue
        if error:
            failed += 1
            continue
        kept.append(row)
    return kept, missing, failed


# 영상 하나 = 전처리한 얼굴 크롭 frames 장. 저장된 순서(시간 순)를 그대로 돌려줍니다.
#   augment      : utils/transition_augment.TransitionAugment. 주면 샘플마다 확률적으로 장면 전환을 넣습니다.
#                  학습에만 씁니다. 평가에는 주지 마세요(실제로 합성한 벤치마크 클립으로 평가합니다).
#   with_thumbs  : 배경 썸네일도 함께 돌려줍니다. 전환 증강은 얼굴과 썸네일에 같은 전환을 적용합니다.
class FaceClipDataset(Dataset):
    def __init__(self, rows, npz_root, frames=30, augment=None, with_thumbs=False):
        self.rows = rows
        self.npz_root = Path(npz_root)
        self.frames = frames
        self.augment = augment
        self.with_thumbs = with_thumbs

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        keys = ("faces", "thumbs") if self.with_thumbs else ("faces",)
        with np.load(self.npz_root / f"{row['video_id']}.npz") as data:
            sample = {key: data[key][:self.frames] for key in keys}
        if self.augment is not None:
            sample, _ = self.augment(sample, row)
        # torch.from_numpy: 복사 없이 같은 메모리를 텐서로 씁니다.
        faces = torch.from_numpy(np.ascontiguousarray(sample["faces"]))
        if not self.with_thumbs:
            return faces, row["label"], row["video_id"]
        thumbs = torch.from_numpy(np.ascontiguousarray(sample["thumbs"]))
        return faces, thumbs, row["label"], row["video_id"]


# 원 논문 Table 4 의 학습 순서를 만드는 배치 샘플러입니다.
#   논문: real N개, fake M개(N < M). i 번째 real 과 (N x c + i) 번째 fake 를 한 배치(크기 2)로 묶고,
#         real 목록을 처음부터 다시 돌면서(c = 0, 1, ...) fake 를 모두 한 번씩 쓰면 끝납니다.
#         Celeb-DF 학습 분할은 real 490 / fake 5,539 라서, 한 번 돌 때 real 은 약 11번씩 쓰입니다.
#   여기서 1 에폭 = fake 를 한 번씩 모두 쓰는 것 = len(fake) // pairs_per_batch 스텝입니다.
#   논문에 없는 부분: 파일 이름 순서(같은 인물끼리 몰려 있음)로 학습하지 않도록, 에폭마다 real 과 fake 순서를 섞습니다.
#   real 은 한 바퀴마다 새로 섞어 이어 붙이므로, 어떤 real 도 다른 real 보다 1번 넘게 더 쓰이지 않습니다.
class PairedBatchSampler(Sampler):
    def __init__(self, labels, pairs_per_batch=1, seed=0):
        super().__init__()
        labels = list(labels)
        self.real = [i for i, label in enumerate(labels) if label == 0]
        self.fake = [i for i, label in enumerate(labels) if label == 1]
        if not self.real or not self.fake:
            raise ValueError("real 과 fake 영상이 모두 있어야 합니다.")
        self.pairs_per_batch = pairs_per_batch
        self.seed = seed
        self.epoch = 0

    # 에폭마다 다른 순서가 나오도록 학습 코드가 매 에폭 시작 전에 부릅니다.
    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.fake) // self.pairs_per_batch

    def __iter__(self):
        # [seed, epoch] 로 난수 생성기를 만들면, 같은 seed·에폭에서는 언제 실행해도 같은 순서가 나옵니다.
        rng = np.random.default_rng([self.seed, self.epoch])
        fake = rng.permutation(self.fake).tolist()
        real = []
        while len(real) < len(fake):
            real += rng.permutation(self.real).tolist()

        size = self.pairs_per_batch
        for batch in range(len(self)):
            start = batch * size
            yield real[start:start + size] + fake[start:start + size]
