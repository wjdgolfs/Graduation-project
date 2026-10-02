# ============================================================================
# 학습과 평가: 원 논문 재현 모델(1단계)과 배경 기준 정규화 모델(5단계)
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
# 5단계 기준선과 제안 모델 (docs/04_5단계_설계.md)
#   B0  python scripts/train_paper.py --dataset ffpp --run-name b0_ffpp
#   B1  python scripts/train_paper.py --dataset ffpp --run-name b1_augment --augment
#   P   python scripts/train_paper.py --dataset ffpp --run-name p_normalized --augment --model normalized
#   --model normalized 이면 배경 썸네일을 함께 읽어 모델에 넣고, 예측 CSV 에 영상별 평균 α 를 적습니다.
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

from models.normalized_bclstm import build_normalized_model  # noqa: E402
from models.paper_bclstm import build_model, prepare_clips  # noqa: E402
from utils.face_clips import ClipLoader, FaceClipDataset, PairedBatchSampler, load_index, usable_rows  # noqa: E402
from utils.metrics import classification_metrics, format_metrics  # noqa: E402
from utils.transition_augment import build_augment  # noqa: E402


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


# 배치에서 (얼굴, 썸네일 또는 None, 라벨, video_id) 를 꺼냅니다. 제안 모델은 썸네일이 하나 더 붙습니다.
def split_batch(batch):
    if len(batch) == 4:
        return batch
    faces, labels, video_ids = batch
    return faces, None, labels, video_ids


# 한 배치를 모델에 넣습니다. 반환값: (로짓, α 또는 None)
# 제안 모델만 썸네일을 받고 α 를 돌려줍니다.
def forward_batch(model, faces, thumbs, device, amp, return_alpha=False):
    clips = prepare_clips(faces.to(device, non_blocking=True))
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
        if thumbs is None:
            return model(clips), None
        if return_alpha:
            return model(clips, thumbs.to(device, non_blocking=True), return_alpha=True)
        return model(clips, thumbs.to(device, non_blocking=True)), None


# 테스트 영상 전체의 fake 확률을 구합니다. 반환값: (video_id 목록, 라벨 목록, fake 확률 목록, 영상별 평균 α 목록 또는 None)
# @torch.no_grad(): 이 함수 안에서는 기울기를 계산하지 않아 메모리와 시간을 아낍니다.
@torch.no_grad()
def predict(model, loader, device, amp, max_batches=None):
    model.eval()
    video_ids, labels, probabilities, alphas = [], [], [], []
    for index, batch in enumerate(tqdm(loader, desc="test", leave=False), start=1):
        faces, thumbs, batch_labels, batch_ids = split_batch(batch)
        logits, alpha = forward_batch(model, faces, thumbs, device, amp, return_alpha=True)
        # softmax 로 두 로짓을 합이 1 인 확률로 바꾸고, 1번(fake) 확률만 모읍니다.
        probabilities += torch.softmax(logits.float(), dim=1)[:, 1].tolist()
        labels += batch_labels.tolist()
        video_ids += list(batch_ids)
        if alpha is not None:
            # 영상마다 프레임 평균 하나로 줄입니다. 프레임별 값은 학습이 끝난 뒤 따로 분석합니다.
            alphas += alpha.float().mean(dim=1).tolist()
        if max_batches is not None and index >= max_batches:
            break
    model.train()
    return video_ids, labels, probabilities, (alphas or None)


def write_predictions(path, video_ids, labels, probabilities, alphas=None):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        columns = ["video_id", "label", "prob_fake", "prediction"] + (["alpha_mean"] if alphas else [])
        writer.writerow(columns)
        for index, (video_id, label, probability) in enumerate(zip(video_ids, labels, probabilities)):
            row = [video_id, label, f"{probability:.6f}", int(probability >= 0.5)]
            if alphas:
                row.append(f"{alphas[index]:.4f}")
            writer.writerow(row)


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
    parser.add_argument("--max-eval", type=int, default=None, help="테스트를 이 배치 수까지만 합니다(시험용).")
    parser.add_argument("--no-pretrained", action="store_true", help="XceptionNet 을 ImageNet 가중치 없이 시작합니다(시험용).")
    parser.add_argument("--model", choices=["paper", "normalized"], default="paper",
                        help="paper: 원 논문 모델 / normalized: 5단계 제안 모델(배경 분기 + 학습되는 α)")
    parser.add_argument("--dataset", default=None, help="학습에 쓸 데이터셋 폴더 이름 (기본: paper_model.dataset)")
    parser.add_argument("--epochs", type=int, default=None, help="에폭 수 (기본: paper_model.epochs)")
    parser.add_argument("--augment", dest="augment", action="store_true", default=None,
                        help="전환 증강을 켭니다(기준선 B1, 제안 모델 학습).")
    parser.add_argument("--no-augment", dest="augment", action="store_false",
                        help="설정에서 켜져 있어도 전환 증강을 끕니다.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU 가 필요합니다.")
    config = load_config()
    settings = dict(config["paper_model"])
    extra = config["normalized_model"]
    # 명령줄로 준 값이 설정값을 덮습니다. 체크포인트에도 이 settings 가 그대로 저장됩니다.
    if args.dataset:
        settings["dataset"] = args.dataset
    if args.epochs:
        settings["epochs"] = args.epochs
    settings["model_kind"] = args.model
    # 증강은 --augment / --no-augment 가 없으면 설정값(transition_augment.enabled)을 따릅니다.
    augment_settings = dict(config["transition_augment"])
    if args.augment is not None:
        augment_settings["enabled"] = args.augment
    settings["transition_augment"] = augment_settings
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
    # 영상 하나를 어떻게 읽을지 정하는 객체. 학습·평가·전환 증강이 모두 같은 것을 씁니다.
    # 제안 모델은 배경 썸네일도 필요합니다(얼굴 자리를 가리고 고정 크기로 맞춰 읽습니다).
    clip_loader = ClipLoader(
        npz_root,
        settings["frames"],
        with_thumbs=args.model == "normalized",
        thumb_size=extra["thumb_size"],
        exclusion_scale=extra["background_exclusion_scale"],
    )
    # 평가에는 증강을 쓰지 않습니다(전환 강건성은 실제로 합성한 벤치마크 클립으로 잽니다).
    test_loader = DataLoader(
        FaceClipDataset(splits["test"], clip_loader),
        batch_size=2 * settings["pairs_per_batch"],
        shuffle=False,
        **loader_options,
    )

    # 2) 모델. 테스트만 할 때는 체크포인트가 모든 가중치를 덮으므로 ImageNet 가중치를 읽지 않습니다.
    pretrained = not (args.no_pretrained or args.evaluate)
    if args.model == "normalized":
        model = build_normalized_model(settings, extra, pretrained=pretrained).to(device)
    else:
        model = build_model(settings, pretrained=pretrained).to(device)
    checkpoint_dir = Path(settings["checkpoint_dir"]) / args.run_name
    results_dir = Path(settings["results_dir"]) / args.run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.evaluate:
        # weights_only=True: 가중치·숫자·문자열만 읽고, 파일 안의 임의 파이썬 코드는 실행하지 않습니다.
        state = torch.load(args.evaluate, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        video_ids, labels, probabilities, alphas = predict(model, test_loader, device, amp, args.max_eval)
        write_predictions(results_dir / f"predictions_{Path(args.evaluate).stem}.csv",
                          video_ids, labels, probabilities, alphas)
        print(format_metrics(classification_metrics(labels, probabilities)))
        return

    # 3) 학습 준비
    sampler = PairedBatchSampler([row["label"] for row in splits["train"]], settings["pairs_per_batch"], settings["seed"])
    # 전환 증강: 학습 분할 안에서만 짝을 찾습니다(테스트 영상이 학습에 섞이지 않게).
    augment = build_augment(augment_settings, splits["train"], clip_loader)
    if augment is not None:
        kinds = ", ".join(f"{kind} {lengths}" for kind, lengths in augment_settings["transitions"].items())
        print(f"전환 증강 켜짐: 확률 {augment_settings['probability']}, {kinds}")
    train_loader = DataLoader(
        FaceClipDataset(splits["train"], clip_loader, augment=augment),
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
        for step, batch in enumerate(progress, start=1):
            faces, thumbs, labels, _ = split_batch(batch)
            # non_blocking=True: pin_memory 된 텐서를 기다리지 않고 GPU 로 보냅니다.
            # autocast: 합성곱 같은 연산을 bf16(16비트)으로 계산해 메모리와 시간을 줄입니다. 손실은 float32 로 계산합니다.
            labels = labels.to(device, non_blocking=True)
            logits, _alpha = forward_batch(model, faces, thumbs, device, amp)
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
        video_ids, test_labels, probabilities, alphas = predict(model, test_loader, device, amp, args.max_eval)
        metrics = classification_metrics(test_labels, probabilities)
        write_predictions(results_dir / f"predictions_epoch_{epoch + 1:02d}.csv",
                          video_ids, test_labels, probabilities, alphas)
        append_row(results_dir / "metrics.csv", METRIC_COLUMNS, {"epoch": epoch + 1, **metrics, "train_minutes": round(train_minutes, 1)})
        torch.save(
            # settings 에 model_kind 와 증강 설정이 들어 있어, 나중에 체크포인트만 보고 같은 모델을 다시 만들 수 있습니다.
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
             "settings": settings, "extra": extra, "metrics": metrics},
            checkpoint_dir / f"epoch_{epoch + 1:02d}.pt",
        )
        print(f"[epoch {epoch + 1}] {format_metrics(metrics)} | 학습 {train_minutes:.1f}분")


# DataLoader 가 자식 프로세스를 쓰므로, 윈도우에서는 반드시 이 조건 안에서 시작해야 합니다.
if __name__ == "__main__":
    main()
