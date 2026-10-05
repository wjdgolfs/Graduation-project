# ============================================================================
# 제안 모델의 프레임별 정규화 강도 α 를 뽑아 저장합니다 (논문 그림 6 의 재료)
# ----------------------------------------------------------------------------
# 묻는 것
#   모델이 전환 구간에서 실제로 더 많이 빼는가? α_t 를 시간축에 그려 전환 구간과 겹쳐 보기 위한 데이터를 만듭니다.
#
# 입력 구간
#   모델 입력이 30프레임이라 전환 중심을 가운데 둔 113~142번 프레임입니다.
#   디졸브 32·64 는 이 구간 전체가 전환 안에 들어가므로, 한 클립 안에서 안/밖을 나눌 수 없습니다.
#   따라서 그림은 "대조군 클립의 α"와 "전환 클립의 α"를 같은 축에 겹쳐 그리는 방식으로 봅니다.
#
# 입력: benchmark.output_dir/faces/<기법>/<클립>.npz, manifest, 제안 모델 체크포인트
# 출력: paths.results/benchmark/alpha_frames.npz
#   alpha      [클립 수, 30]  프레임별 α
#   clip_id, source, kind, label  [클립 수]
#   frames     [30]           원본 클립에서의 프레임 번호
#
# 실행 예 (프로젝트 폴더에서)
#   python scripts/export_alpha.py --checkpoint D:/.../p_normalized/epoch_01.pt
# ============================================================================

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.normalized_bclstm import build_normalized_model  # noqa: E402
from models.paper_bclstm import prepare_clips  # noqa: E402
from utils.face_clips import ClipLoader  # noqa: E402


def load_config():
    with open(PROJECT_ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    config = load_config()
    benchmark = config["benchmark"]
    root = Path(benchmark["output_dir"])

    parser = argparse.ArgumentParser(description="제안 모델의 프레임별 α 를 뽑습니다.")
    parser.add_argument("--checkpoint", required=True, help="--model normalized 로 학습한 체크포인트")
    parser.add_argument("--manifest", default=str(root / "manifest_pass1.csv"), help="클립 목록 CSV")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    settings, extra = state["settings"], state["extra"]
    if settings.get("model_kind") != "normalized":
        raise SystemExit("제안 모델(--model normalized)로 학습한 체크포인트가 필요합니다.")
    model = build_normalized_model(settings, extra, pretrained=False)
    model.load_state_dict(state["model"])
    model = model.to(device).eval()

    with open(args.manifest, newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file)
                if (root / "faces" / row["source"] / f"{row['clip_id']}.npz").exists()]
    print(f"클립 {len(rows)}개의 α 를 뽑습니다.")

    loader = ClipLoader(root / "faces", settings["frames"], with_thumbs=True,
                        thumb_size=extra["thumb_size"], exclusion_scale=extra["background_exclusion_scale"])

    alphas, kept = [], []
    faces_batch, thumbs_batch, batch_rows = [], [], []

    @torch.no_grad()
    def flush():
        if not batch_rows:
            return
        clips = prepare_clips(torch.stack(faces_batch)).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=settings["amp"]):
            _, alpha = model(clips, torch.stack(thumbs_batch).to(device), return_alpha=True)
        alphas.append(alpha.float().cpu().numpy())
        kept.extend(batch_rows)
        faces_batch.clear()
        thumbs_batch.clear()
        batch_rows.clear()

    for row in tqdm(rows, unit="클립"):
        sample = loader(f"{row['source']}/{row['clip_id']}")
        if sample is None:
            continue
        faces_batch.append(torch.from_numpy(np.ascontiguousarray(sample["faces"])))
        thumbs_batch.append(torch.from_numpy(np.ascontiguousarray(sample["thumbs"])))
        batch_rows.append(row)
        if len(batch_rows) >= args.batch_size:
            flush()
    flush()

    start = benchmark["transition_center"] - settings["frames"] // 2
    out_path = Path(config["paths"]["results"]) / "benchmark" / "alpha_frames.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        alpha=np.concatenate(alphas),
        clip_id=np.array([row["clip_id"] for row in kept]),
        source=np.array([row["source"] for row in kept]),
        kind=np.array([row["kind"] + (row["length"] if row["length"] != "0" else "") for row in kept]),
        label=np.array([int(row["label"]) for row in kept]),
        window_start=np.array([int(row["window_start"]) for row in kept]),
        window_end=np.array([int(row["window_end"]) for row in kept]),
        frames=np.arange(start, start + settings["frames"]),
    )
    print(f"저장: {out_path} (클립 {len(kept)}개 x {settings['frames']}프레임)")


if __name__ == "__main__":
    main()
