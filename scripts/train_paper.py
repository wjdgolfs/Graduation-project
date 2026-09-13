# ============================================================================
# 원 논문 재현: XceptionNet + CBAM + Bi-ConvLSTM 학습과 평가 (Celeb-DF)
# ----------------------------------------------------------------------------
# 무엇을 하나
#   1) 얼굴 전처리 결과(index.csv 와 영상별 .npz)에서 논문 방식 분할(split 열)의 train / test 영상을 읽습니다.
#   2) 논문 Table 4 처럼 매 스텝 real 1개 + fake 1개(배치 2)로 학습합니다. fake 를 한 번씩 모두 쓰면 1 에폭입니다.
#   3) 에폭마다 test(real 100 / fake 100)로 정확도·정밀도·재현율·F1·AUC 를 재고, 가중치와 영상별 예측을 저장합니다.
#      논문처럼 테스트 결과로 에폭을 고르지 않습니다. 설정한 마지막 에폭의 결과를 재현 결과로 봅니다.
#
# 실행 예 (프로젝트 폴더에서)
#   python scripts/train_paper.py --max-steps 20 --run-name smoke     몇 스텝만 돌려 전체 흐름 확인
#   python scripts/train_paper.py                                     설정대로 학습
#   python scripts/train_paper.py --resume <체크포인트.pt>             저장된 에폭 다음부터 이어서 학습
#   python scripts/train_paper.py --evaluate <체크포인트.pt>           테스트만
#
# 출력 (<run> = --run-name)
#   paper_model.checkpoint_dir/<run>/epoch_XX.pt         모델·옵티마이저 상태, 설정, 테스트 지표
#   paper_model.results_dir/<run>/train_log.csv          log_every 스텝마다 평균 손실·정확도
#   paper_model.results_dir/<run>/metrics.csv            에폭마다 테스트 지표
#   paper_model.results_dir/<run>/predictions_*.csv      영상별 fake 확률과 판정
# ============================================================================

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm


# __file__는 현재 파일 경로입니다. 상위 두 단계인 프로젝트 폴더를 기준으로 설정 파일을 찾습니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# DataLoader 의 자식 프로세스도 이 파일을 다시 읽으므로 같은 경로가 들어갑니다.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.paper_bclstm import build_model, prepare_clips  # noqa: E402
from utils.face_clips import FaceClipDataset, PairedBatchSampler, load_index, usable_rows  # noqa: E402
from utils.metrics import classification_metrics, format_metrics  # noqa: E402


METRIC_COLUMNS = ["epoch", "accuracy", "precision", "recall", "f1", "auc", "tp", "fp", "tn", "fn", "train_minutes"]


# 여러 스크립트가 같은 경로와 기준값을 쓰도록 YAML 설정을 딕셔너리로 읽습니다.
def load_config():
    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return yaml.safe_load(file)


# 난수 시드를 고정해, 같은 설정이면 같은 초기 가중치·같은 데이터 순서로 시작하게 합니다.
# (GPU 합성곱의 아주 작은 계산 차이까지 없애지는 않습니다.)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# 테스트 영상 전체의 fake 확률을 구합니다. 반환값: (video_id 목록, 라벨 목록, fake 확률 목록)
# @torch.no_grad(): 이 함수 안에서는 기울기를 계산하지 않아 메모리와 시간을 아낍니다.
@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    video_ids, labels, probabilities = [], [], []
    for faces, batch_labels, batch_ids in tqdm(loader, desc="test", leave=False):
        clips = prepare_clips(faces.to(device, non_blocking=True))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(clips)
        # softmax 로 두 로짓을 합이 1 인 확률로 바꾸고, 1번(fake) 확률만 모읍니다.
        probabilities += torch.softmax(logits.float(), dim=1)[:, 1].tolist()
        labels += batch_labels.tolist()
        video_ids += list(batch_ids)
    model.train()
    return video_ids, labels, probabilities


def write_predictions(path, video_ids, labels, probabilities):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["video_id", "label", "prob_fake", "prediction"])
        for video_id, label, probability in zip(video_ids, labels, probabilities):
            writer.writerow([video_id, label, f"{probability:.6f}", int(probability >= 0.5)])


# CSV 파일 끝에 한 줄을 붙입니다. 파일이 없으면 머리글부터 씁니다.
def append_row(path, columns, row):
    is_new = not Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="원 논문(XceptionNet + CBAM + Bi-ConvLSTM) 재현 학습과 평가")
    parser.add_argument("--run-name", default="paper_repro", help="결과·체크포인트 폴더 이름")
    parser.add_argument("--resume", default=None, help="이 체크포인트(.pt)의 다음 에폭부터 이어서 학습합니다.")
    parser.add_argument("--evaluate", default=None, help="이 체크포인트(.pt)로 테스트만 합니다.")
    parser.add_argument("--max-steps", type=int, default=None, help="에폭마다 이 스텝 수까지만 학습합니다(시험용).")
    parser.add_argument("--no-pretrained", action="store_true", help="XceptionNet 을 ImageNet 가중치 없이 시작합니다(시험용).")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU 가 필요합니다.")
    config = load_config()
    settings = config["paper_model"]
    device = torch.device("cuda")
    amp = settings["amp"]
    set_seed(settings["seed"])
    # cudnn.benchmark 는 첫 스텝에서 합성곱 계산 방식을 여러 개 직접 돌려 보고 가장 빠른 것을 고르는 기능입니다.
    # 이 노트북에서 켜고 측정했을 때 첫 스텝이 25~130초로 늘어졌고, 이어서 그래픽 드라이버 시간 초과 블루스크린(0x116)이 났습니다.
    # 원인일 가능성이 있어 끕니다. 끄면 cuDNN 이 미리 정한 규칙으로 방식을 고릅니다.
    torch.backends.cudnn.benchmark = False

    # 1) 데이터 목록. 전처리가 아직 끝나지 않았거나 실패한 영상은 빠지고, 그 수를 보여줍니다.
    npz_root = Path(config["preprocess"]["output_dir"]) / settings["dataset"]
    splits = {}
    for split in ("train", "test"):
        rows, missing, failed = usable_rows(load_index(npz_root / "index.csv", split), npz_root)
        real = sum(row["label"] == 0 for row in rows)
        print(f"[{split}] real {real} / fake {len(rows) - real} (전처리 결과 없음 {missing}, 실패 {failed})")
        splits[split] = rows

    # pin_memory: CPU 쪽 텐서를 GPU 로 빨리 옮길 수 있는 메모리에 둡니다.
    # persistent_workers: 에폭이 끝나도 자식 프로세스를 유지해 다시 띄우는 시간을 아낍니다.
    loader_options = {
        "num_workers": settings["num_workers"],
        "pin_memory": True,
        "persistent_workers": settings["num_workers"] > 0,
    }
    test_loader = DataLoader(
        FaceClipDataset(splits["test"], npz_root, settings["frames"]),
        batch_size=2 * settings["pairs_per_batch"],
        shuffle=False,
        **loader_options,
    )

    # 2) 모델. 테스트만 할 때는 체크포인트가 모든 가중치를 덮으므로 ImageNet 가중치를 읽지 않습니다.
    model = build_model(settings, pretrained=not (args.no_pretrained or args.evaluate)).to(device)
    checkpoint_dir = Path(settings["checkpoint_dir"]) / args.run_name
    results_dir = Path(settings["results_dir"]) / args.run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.evaluate:
        # weights_only=True: 가중치·숫자·문자열만 읽고, 파일 안의 임의 파이썬 코드는 실행하지 않습니다.
        state = torch.load(args.evaluate, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        video_ids, labels, probabilities = predict(model, test_loader, device, amp)
        write_predictions(results_dir / f"predictions_{Path(args.evaluate).stem}.csv", video_ids, labels, probabilities)
        print(format_metrics(classification_metrics(labels, probabilities)))
        return

    # 3) 학습 준비
    sampler = PairedBatchSampler([row["label"] for row in splits["train"]], settings["pairs_per_batch"], settings["seed"])
    train_loader = DataLoader(
        FaceClipDataset(splits["train"], npz_root, settings["frames"]),
        batch_sampler=sampler,
        **loader_options,
    )
    # Adam 과 학습률·β 값은 논문 값입니다.
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"], betas=tuple(settings["betas"]))
    # CrossEntropyLoss = softmax + 교차 엔트로피 E = -Σ t_k log y_k (논문 식 13)
    criterion = torch.nn.CrossEntropyLoss()

    start_epoch = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        print(f"{args.resume} 에서 이어서 {start_epoch + 1} 에폭부터 학습합니다.")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 4) 학습
    steps_per_epoch = min(len(sampler), args.max_steps or len(sampler))
    for epoch in range(start_epoch, settings["epochs"]):
        sampler.set_epoch(epoch)
        model.train()
        started = time.time()
        running_loss = running_correct = running_count = 0
        progress = tqdm(train_loader, total=steps_per_epoch, desc=f"epoch {epoch + 1}/{settings['epochs']}")
        for step, (faces, labels, _) in enumerate(progress, start=1):
            # non_blocking=True: pin_memory 된 텐서를 기다리지 않고 GPU 로 보냅니다.
            clips = prepare_clips(faces.to(device, non_blocking=True))
            labels = labels.to(device, non_blocking=True)
            # autocast: 합성곱 같은 연산을 bf16(16비트)으로 계산해 메모리와 시간을 줄입니다. 손실은 float32 로 계산합니다.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(clips)
            loss = criterion(logits.float(), labels)

            # 기울기 초기화 → 역전파로 기울기 계산 → 가중치 갱신
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(labels)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_count += len(labels)
            if step % settings["log_every"] == 0 or step == steps_per_epoch:
                row = {
                    "epoch": epoch + 1,
                    "step": step,
                    "loss": round(running_loss / running_count, 5),
                    "accuracy": round(running_correct / running_count, 4),
                    "minutes": round((time.time() - started) / 60, 2),
                }
                append_row(results_dir / "train_log.csv", list(row), row)
                progress.set_postfix(loss=row["loss"], acc=row["accuracy"])
                running_loss = running_correct = running_count = 0
            if step >= steps_per_epoch:
                break
        train_minutes = (time.time() - started) / 60

        # 5) 에폭마다 테스트하고 저장
        video_ids, test_labels, probabilities = predict(model, test_loader, device, amp)
        metrics = classification_metrics(test_labels, probabilities)
        write_predictions(results_dir / f"predictions_epoch_{epoch + 1:02d}.csv", video_ids, test_labels, probabilities)
        append_row(results_dir / "metrics.csv", METRIC_COLUMNS, {"epoch": epoch + 1, **metrics, "train_minutes": round(train_minutes, 1)})
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "settings": settings, "metrics": metrics},
            checkpoint_dir / f"epoch_{epoch + 1:02d}.pt",
        )
        print(f"[epoch {epoch + 1}] {format_metrics(metrics)} | 학습 {train_minutes:.1f}분")


# DataLoader 가 자식 프로세스를 쓰므로, 윈도우에서는 반드시 이 조건 안에서 시작해야 합니다.
if __name__ == "__main__":
    main()
