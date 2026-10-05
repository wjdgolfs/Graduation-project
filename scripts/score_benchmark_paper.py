# ============================================================================
# 학습한 모델로 전환 벤치마크를 채점합니다 (5단계 평가, 6단계 일반성 확인)
# ----------------------------------------------------------------------------
# 묻는 것
#   전환이 탐지 점수를 얼마나 흔드는가를 모델별로 비교합니다.
#   같은 기준 영상에서 만든 대조군과 전환 합성본을 짝지어, logit 변화와 가짜 판정 비율을 봅니다.
#     real 클립의 점수가 오르면 → 오탐(장면 전환을 조작으로 착각)
#     fake 클립의 점수가 내리면 → 미탐(장면 전환이 조작을 가림)
#
# 입력 구간
#   모델 입력이 30프레임이라 전환 중심을 가운데 둔 113~142번 프레임을 씁니다
#   (preprocessing/benchmark_faces.py 가 미리 뽑아 둔 얼굴 크롭).
#   디졸브 32·64 는 이 구간 전체가 전환 구간 안에 들어갑니다. 그래서 한 클립 안에서 전환 안/밖을 나눌 수 없고,
#   대조군 클립과 비교하는 방식만 씁니다.
#   FTCN 은 32프레임 클립을 겹쳐 가며 영상 전체를 보므로 보는 범위가 다릅니다.
#   두 모델의 점수를 직접 비교하지 않고, 각각 "대조군 대비 변화량"으로 비교합니다.
#
# 제안 모델(--model normalized 로 학습한 체크포인트)이면 배경 썸네일도 함께 넣고,
# 프레임마다의 정규화 강도 α 의 평균을 클립별로 기록합니다(전환 구간에서 α 가 커지는지 보기 위해).
#
# 입력: benchmark.output_dir/faces/<기법>/<클립>.npz, manifest.csv, 학습한 체크포인트(.pt)
# 출력: 화면 표 + paths.results/benchmark/effect_<체크포인트 폴더 이름>.csv
#
# 실행 예 (프로젝트 폴더에서)
#   python scripts/score_benchmark_paper.py --checkpoint D:/.../checkpoints/paper_bclstm/b0_ffpp/epoch_01.pt
# ============================================================================

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import wilcoxon
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.normalized_bclstm import build_normalized_model  # noqa: E402
from models.paper_bclstm import build_model, prepare_clips  # noqa: E402
from utils.face_clips import ClipLoader  # noqa: E402

# 모델은 2-클래스 softmax 라 판정 기준은 0.5 입니다(학습 때와 같은 규칙).
DECISION_THRESHOLD = 0.5
# logit 변환에서 0 과 1 을 피하기 위한 여유값
EPS = 1e-6


def load_config():
    with open(PROJECT_ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# 확률을 logit 으로 바꿉니다. 점수가 0 이나 1 에 몰려 있어 그대로 빼면 차이가 묻히기 때문입니다(0단계와 같은 처리).
def logit(probability):
    clipped = min(max(float(probability), EPS), 1.0 - EPS)
    return float(np.log(clipped / (1.0 - clipped)))


# 체크포인트에 저장된 설정으로 모델을 만들고 가중치를 올립니다. 반환값: (모델, 설정, 제안 모델인지)
def load_model(path, config, device):
    # weights_only=True: 가중치·숫자·문자열만 읽고 파일 안의 임의 코드는 실행하지 않습니다.
    state = torch.load(path, map_location="cpu", weights_only=True)
    settings = state.get("settings", config["paper_model"])
    extra = state.get("extra", config["normalized_model"])
    normalized = settings.get("model_kind") == "normalized"
    if normalized:
        model = build_normalized_model(settings, extra, pretrained=False)
    else:
        model = build_model(settings, pretrained=False)
    model.load_state_dict(state["model"])
    return model.to(device).eval(), settings, extra, normalized


# 클립을 batch_size 개씩 묶어 추론합니다. 반환값: {(기법, 클립): (fake 확률, α 평균 또는 nan)}
@torch.no_grad()
def score_clips(model, loader, jobs, device, amp, batch_size, normalized):
    scores = {}
    batch_faces, batch_thumbs, keys = [], [], []

    def flush():
        if not keys:
            return
        clips = prepare_clips(torch.stack(batch_faces)).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            if normalized:
                logits, alpha = model(clips, torch.stack(batch_thumbs).to(device), return_alpha=True)
                alphas = alpha.float().mean(dim=1).tolist()
            else:
                logits = model(clips)
                alphas = [float("nan")] * len(keys)
            probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
        for key, probability, mean_alpha in zip(keys, probabilities.tolist(), alphas):
            scores[key] = (probability, mean_alpha)
        batch_faces.clear()
        batch_thumbs.clear()
        keys.clear()

    for source, clip_id in tqdm(jobs, unit="클립"):
        sample = loader(f"{source}/{clip_id}")
        if sample is None:
            continue
        batch_faces.append(torch.from_numpy(np.ascontiguousarray(sample["faces"])))
        if normalized:
            batch_thumbs.append(torch.from_numpy(np.ascontiguousarray(sample["thumbs"])))
        keys.append((source, clip_id))
        if len(keys) >= batch_size:
            flush()
    flush()
    return scores


# 짝지은 차이에 부호순위 검정을 합니다(0 보다 큰 쪽으로). 표본이 적으면 nan 입니다.
def wilcoxon_p(differences):
    values = np.asarray([value for value in differences if np.isfinite(value) and value != 0])
    if len(values) < 6:
        return float("nan")
    return float(wilcoxon(values, alternative="greater").pvalue)


def mean_or_nan(values):
    values = [value for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def main():
    config = load_config()
    benchmark = config["benchmark"]
    root = Path(benchmark["output_dir"])

    parser = argparse.ArgumentParser(description="학습한 모델로 벤치마크를 채점해 전환 효과를 봅니다.")
    parser.add_argument("--checkpoint", required=True, help="scripts/train_paper.py 가 저장한 .pt")
    parser.add_argument("--manifest", default=str(root / "manifest.csv"), help="클립 목록 CSV")
    parser.add_argument("--batch-size", type=int, default=4, help="한 번에 추론할 클립 수")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cudnn.benchmark 를 켠 상태로 돌리다 드라이버 시간 초과 블루스크린(0x116)이 난 적이 있습니다. 끈 채로 둡니다.
    torch.backends.cudnn.benchmark = False

    model, settings, extra, normalized = load_model(args.checkpoint, config, device)
    run_name = Path(args.checkpoint).parent.name
    print(f"체크포인트 {run_name}/{Path(args.checkpoint).name} | 학습 {settings.get('dataset', '?')} | "
          f"모델 {'제안(정규화)' if normalized else '원 논문'} | "
          f"전환 증강 {settings.get('transition_augment', {}).get('enabled', '?')}")

    with open(args.manifest, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    loader = ClipLoader(
        root / "faces",
        settings["frames"],
        with_thumbs=normalized,
        thumb_size=extra["thumb_size"],
        exclusion_scale=extra["background_exclusion_scale"],
    )
    jobs = [(row["source"], row["clip_id"]) for row in rows
            if (root / "faces" / row["source"] / f"{row['clip_id']}.npz").exists()]
    print(f"클립 {len(jobs)}개를 채점합니다 (얼굴 크롭이 없는 클립 {len(rows) - len(jobs)}개는 건너뜀).")
    if not jobs:
        print("preprocessing/benchmark_faces.py 를 먼저 돌리세요.")
        return

    scores = score_clips(model, loader, jobs, device, settings["amp"], args.batch_size, normalized)

    # 기준 영상마다 대조군과 전환 클립을 모읍니다.
    controls, transitions = {}, defaultdict(list)
    for row in rows:
        key = (row["source"], row["base_id"])
        if row["kind"] == "control":
            controls[key] = row
        else:
            transitions[key].append(row)

    records = []
    for key, group in transitions.items():
        control_row = controls.get(key)
        if control_row is None:
            continue
        control_key = (control_row["source"], control_row["clip_id"])
        if control_key not in scores:
            continue
        control_score, control_alpha = scores[control_key]
        for row in group:
            clip_key = (row["source"], row["clip_id"])
            if clip_key not in scores:
                continue
            score, alpha = scores[clip_key]
            record = {
                "clip_id": row["clip_id"],
                "source": row["source"],
                "label": int(row["label"]),
                "kind": row["kind"] + (row["length"] if row["length"] != "0" else ""),
                "control_score": control_score,
                "score": score,
                "control_logit": logit(control_score),
                "clip_logit": logit(score),
                "control_alpha": control_alpha,
                "alpha": alpha,
            }
            record["delta_logit"] = record["clip_logit"] - record["control_logit"]
            record["delta_alpha"] = alpha - control_alpha
            records.append(record)

    if not records:
        print("짝지을 클립이 없습니다.")
        return

    print(f"\n짝지어 비교한 클립 {len(records)}개 (판정 기준 {DECISION_THRESHOLD})\n")
    header = ("전환", "정답", "쌍", "logit 변화", "오른 비율", "p값", "가짜 판정", "α 변화")
    print(f"{header[0]:<12}{header[1]:<6}{header[2]:>4}{header[3]:>13}{header[4]:>11}{header[5]:>9}"
          f"{header[6]:>18}{header[7]:>10}")
    for label_value, label_name in ((0, "real"), (1, "fake")):
        for kind in sorted({record["kind"] for record in records}):
            group = [record for record in records if record["label"] == label_value and record["kind"] == kind]
            if not group:
                continue
            deltas = [record["delta_logit"] for record in group]
            rose = np.mean([delta > 0 for delta in deltas])
            control_alarm = np.mean([record["control_score"] >= DECISION_THRESHOLD for record in group])
            clip_alarm = np.mean([record["score"] >= DECISION_THRESHOLD for record in group])
            alpha_shift = mean_or_nan([record["delta_alpha"] for record in group])
            print(f"{kind:<12}{label_name:<6}{len(group):>4}{np.median(deltas):>+13.3f}"
                  f"{rose:>10.0%}{wilcoxon_p(deltas):>9.4f}"
                  f"{control_alarm:>12.0%} → {clip_alarm:>3.0%}{alpha_shift:>+10.3f}")

    out_path = Path(config["paths"]["results"]) / "benchmark" / f"effect_{run_name}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\n클립별 값 저장: {out_path}")


if __name__ == "__main__":
    main()
