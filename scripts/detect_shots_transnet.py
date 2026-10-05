# ============================================================================
# 4단계 보강: 전용 신경망 샷 검출기(TransNetV2)로 벤치마크 전환을 찾아 봅니다
# ----------------------------------------------------------------------------
# 왜 필요한가
#   scripts/detect_shots_benchmark.py 는 내용 기반 검출기(PySceneDetect)로 쟀습니다. 그 결과 긴 디졸브를
#   기본 설정에서 0% 잡았습니다. 그러나 TransNetV2 는 점진적 전환을 겨냥해 설계된 전용 신경망으로,
#   학습에 쓰는 합성 전환의 절반을 디졸브로 구성했다고 보고합니다.
#   "기존 해법이 실패한다"는 주장의 범위를 정확히 하려면 이 모델로도 재야 합니다.
#
# 모델
#   공식 저장소(third_party/TransNetV2)의 PyTorch 구현을 쓰고, 공개된 TensorFlow 가중치를 변환해 씁니다.
#   입력은 48x27 RGB 프레임이며, 공식 추론 방식대로 100프레임 창을 50프레임씩 겹쳐 가며 가운데 50프레임의
#   예측만 사용합니다.
#
# 판정 규칙 (PySceneDetect 비교와 동일)
#   전환 구간 [시작, 끝) 의 앞뒤 tolerance(8) 프레임 안에 경계가 하나라도 있으면 "검출"로 봅니다.
#   전환이 없는 대조군에서 경계를 하나라도 찾으면 오검출입니다.
#
# 입력: benchmark.output_dir/manifest.csv 와 그 안의 클립 mp4, 변환한 가중치(.pth)
# 출력: 화면 표 + paths.results/benchmark/shot_detection_transnet.csv
#
# 실행 예 (프로젝트 폴더에서)
#   python scripts/detect_shots_transnet.py --weights D:/Graduation-project-data/checkpoints/transnetv2/transnetv2-pytorch-weights.pth
# ============================================================================

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRANSNET_DIR = PROJECT_ROOT / "third_party" / "TransNetV2" / "inference-pytorch"

# 모델 입력 크기(공식 구현 값)
WIDTH, HEIGHT = 48, 27
WINDOW, STRIDE, MARGIN = 100, 50, 25
TOLERANCE = 8


def load_config():
    with open(PROJECT_ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# 영상을 48x27 RGB 프레임 배열로 읽습니다.
def read_frames(path, limit=None):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while limit is None or len(frames) < limit:
        ok, frame = capture.read()
        if not ok:
            break
        small = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    capture.release()
    return np.asarray(frames, dtype=np.uint8)


# 공식 추론 방식: 앞뒤를 첫·끝 프레임으로 채우고 100프레임 창을 50씩 밀며 가운데 50프레임만 씁니다.
def windows(frames):
    tail = MARGIN + STRIDE - (len(frames) % STRIDE if len(frames) % STRIDE else STRIDE)
    padded = np.concatenate(
        [np.repeat(frames[:1], MARGIN, axis=0), frames, np.repeat(frames[-1:], tail, axis=0)], axis=0)
    start = 0
    while start + WINDOW <= len(padded):
        yield padded[start:start + WINDOW]
        start += STRIDE


@torch.no_grad()
def predict(model, frames, device):
    scores = []
    for window in windows(frames):
        batch = torch.from_numpy(window).unsqueeze(0).to(device)
        single, _ = model(batch)
        scores.append(torch.sigmoid(single)[0, MARGIN:MARGIN + STRIDE, 0].cpu().numpy())
    return np.concatenate(scores)[:len(frames)]


# 예측값이 임계값을 넘는 구간의 대표 프레임(구간의 가운데)을 경계로 봅니다.
def boundaries(scores, threshold):
    found, start = [], None
    for index, value in enumerate(scores):
        if value >= threshold and start is None:
            start = index
        elif value < threshold and start is not None:
            found.append((start + index - 1) // 2)
            start = None
    if start is not None:
        found.append((start + len(scores) - 1) // 2)
    return found


def detected(found, row):
    start, end = int(row["window_start"]), int(row["window_end"])
    return any(start - TOLERANCE <= frame <= end + TOLERANCE for frame in found)


def main():
    config = load_config()
    benchmark = config["benchmark"]
    root = Path(benchmark["output_dir"])

    parser = argparse.ArgumentParser(description="TransNetV2 로 벤치마크의 전환을 검출합니다.")
    parser.add_argument("--weights", required=True, help="변환한 PyTorch 가중치(.pth)")
    parser.add_argument("--manifest", default=str(root / "manifest.csv"))
    parser.add_argument("--threshold", type=float, default=0.5, help="경계로 볼 예측값 기준(공식 기본값 0.5)")
    parser.add_argument("--limit", type=int, default=None, help="앞에서부터 이 개수만 처리합니다(시험용).")
    args = parser.parse_args()

    if not TRANSNET_DIR.exists():
        raise SystemExit(f"TransNetV2 코드가 없습니다: {TRANSNET_DIR}\n"
                         "git clone https://github.com/soCzech/TransNetV2 third_party/TransNetV2")
    sys.path.insert(0, str(TRANSNET_DIR))
    import transnetv2_pytorch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    model = transnetv2_pytorch.TransNetV2()
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    model = model.to(device).eval()

    with open(args.manifest, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if args.limit:
        rows = rows[:args.limit]
    print(f"클립 {len(rows)}개를 검출합니다 (임계값 {args.threshold}, 허용 오차 {TOLERANCE}프레임).")

    records, started = [], time.time()
    for row in tqdm(rows, unit="클립"):
        frames = read_frames(row["path"], benchmark["segment_frames"])
        if len(frames) < 2:
            continue
        scores = predict(model, frames, device)
        found = boundaries(scores, args.threshold)
        records.append({
            "clip_id": row["clip_id"],
            "source": row["source"],
            "label": row["label"],
            "kind": row["kind"],
            "length": row["length"],
            "window_start": row["window_start"],
            "window_end": row["window_end"],
            "boundaries": " ".join(str(frame) for frame in found),
            "detected": int(detected(found, row)) if row["kind"] != "control" else "",
            "false_alarm": int(bool(found)) if row["kind"] == "control" else "",
            "max_score": round(float(scores.max()), 4),
        })

    out_path = Path(config["paths"]["results"]) / "benchmark" / "shot_detection_transnet.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(f"\n전체 {(time.time() - started) / 60:.1f}분\n")
    print(f"{'전환':<16}{'클립':>5}{'검출률':>9}")
    kinds = sorted({(record["kind"], record["length"]) for record in records},
                   key=lambda item: (item[0], int(item[1])))
    for kind, length in kinds:
        group = [record for record in records if record["kind"] == kind and record["length"] == length]
        name = kind + (length if length != "0" else "")
        if kind == "control":
            rate = np.mean([record["false_alarm"] for record in group])
            print(f"{'대조군 오검출':<16}{len(group):>5}{rate:>9.0%}")
        else:
            rate = np.mean([record["detected"] for record in group])
            print(f"{name:<16}{len(group):>5}{rate:>9.0%}")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
