# ============================================================================
# 6단계: 원 논문 모델도 장면 전환에 속는가
# ----------------------------------------------------------------------------
# 묻는 것
#   전환 때문에 생기는 오탐이 FTCN 고유의 문제인지, 시간 정보를 쓰는 탐지기의 공통 문제인지 봅니다.
#   같은 벤치마크 클립을 원 논문 모델(Xception + CBAM + Bi-ConvLSTM, 1단계 재현)로 채점해
#   같은 기준 영상의 대조군과 짝지어 비교합니다.
#
# 입력 구간
#   모델 입력이 30프레임이라 전환 중심을 가운데 둔 113~142번 프레임을 씁니다
#   (preprocessing/benchmark_faces.py 가 미리 뽑아 둔 얼굴 크롭).
#   FTCN 은 32프레임 클립을 겹쳐 가며 영상 전체를 보므로 보는 범위가 다릅니다.
#   그래서 두 모델의 점수를 직접 비교하지 않고, 각각 "대조군 대비 변화량"으로 비교합니다.
#
# 해석 주의
#   1단계 체크포인트는 Celeb-DF 로 학습했습니다. FF++ 벤치마크를 채점하면 교차 데이터셋 평가가 되어
#   절대 정확도는 낮습니다. 그래도 같은 기준 영상의 대조군과 짝지어 보기 때문에 전환 효과는 분리됩니다
#   (영상·인물·기법이 짝 안에서 같고 전환만 다릅니다).
#
# 입력: benchmark.output_dir/faces/<기법>/<클립>.npz, manifest.csv,
#       paper_model.checkpoint_dir/<run>/epoch_XX.pt
# 출력: 화면 표 + paths.results/benchmark/paper_effect.csv (클립마다 점수와 대조군 대비 변화)
#
# 실행 예 (프로젝트 폴더에서)
#   python scripts/score_benchmark_paper.py --checkpoint D:/Graduation-project-data/checkpoints/paper_bclstm/paper_repro/epoch_01.pt
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

from models.paper_bclstm import build_model, prepare_clips  # noqa: E402

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


# 클립 하나의 얼굴 크롭을 읽습니다. 얼굴을 못 찾은 프레임도 모델이 본 그대로 두고 비율만 기록합니다.
def load_clip(path, frames):
    with np.load(path) as data:
        if str(data["error"]).strip():
            return None, 0.0
        faces = data["faces"]
        found = float(np.mean(data["face_found"]))
    if len(faces) < frames:
        return None, found
    return torch.from_numpy(np.ascontiguousarray(faces[:frames])), found


# 클립을 batch_size 개씩 묶어 추론합니다. 반환값: {(기법, 클립): fake 확률}
@torch.no_grad()
def score_clips(model, jobs, frames, device, amp, batch_size):
    scores, found_rates = {}, {}
    batch, keys = [], []

    def flush():
        if not batch:
            return
        clips = prepare_clips(torch.stack(batch)).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            probabilities = torch.softmax(model(clips).float(), dim=1)[:, 1]
        for key, probability in zip(keys, probabilities.tolist()):
            scores[key] = probability
        batch.clear()
        keys.clear()

    for key, path in tqdm(jobs, unit="클립"):
        faces, found = load_clip(path, frames)
        found_rates[key] = found
        if faces is None:
            continue
        batch.append(faces)
        keys.append(key)
        if len(batch) >= batch_size:
            flush()
    flush()
    return scores, found_rates


# 짝지은 차이에 부호순위 검정을 합니다(0 보다 큰 쪽으로). 표본이 적으면 nan 입니다.
def wilcoxon_p(differences):
    values = np.asarray([value for value in differences if np.isfinite(value) and value != 0])
    if len(values) < 6:
        return float("nan")
    return float(wilcoxon(values, alternative="greater").pvalue)


def main():
    config = load_config()
    settings = config["paper_model"]
    benchmark = config["benchmark"]
    root = Path(benchmark["output_dir"])

    parser = argparse.ArgumentParser(description="벤치마크 클립을 원 논문 모델로 채점해 전환 효과를 봅니다.")
    parser.add_argument("--checkpoint", required=True, help="scripts/train_paper.py 가 저장한 .pt")
    parser.add_argument("--manifest", default=str(root / "manifest.csv"), help="클립 목록 CSV")
    parser.add_argument("--batch-size", type=int, default=2, help="한 번에 추론할 클립 수")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cudnn.benchmark 를 켠 상태로 돌리다 드라이버 시간 초과 블루스크린(0x116)이 난 적이 있습니다. 끈 채로 둡니다.
    torch.backends.cudnn.benchmark = False

    # weights_only=True: 가중치·숫자·문자열만 읽고 파일 안의 임의 코드는 실행하지 않습니다.
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    trained = state.get("settings", settings)
    model = build_model(trained, pretrained=False).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    frames = int(trained["frames"])
    print(f"체크포인트 {Path(args.checkpoint).name} | 학습 데이터셋 {trained.get('dataset', '?')} | "
          f"입력 {frames}프레임 x {trained.get('image_size', '?')}px")

    with open(args.manifest, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    faces_root = root / "faces"
    jobs, missing = [], 0
    for row in rows:
        path = faces_root / row["source"] / f"{row['clip_id']}.npz"
        if path.exists():
            jobs.append(((row["source"], row["clip_id"]), path))
        else:
            missing += 1
    print(f"클립 {len(jobs)}개를 채점합니다 (얼굴 크롭이 없는 클립 {missing}개는 건너뜀).")
    if not jobs:
        print("preprocessing/benchmark_faces.py 를 먼저 돌리세요.")
        return

    scores, found_rates = score_clips(model, jobs, frames, device, settings["amp"], args.batch_size)

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
        for row in group:
            clip_key = (row["source"], row["clip_id"])
            if clip_key not in scores:
                continue
            record = {
                "clip_id": row["clip_id"],
                "source": row["source"],
                "label": int(row["label"]),
                "kind": row["kind"] + (row["length"] if row["length"] != "0" else ""),
                "face_found": round(found_rates.get(clip_key, float("nan")), 3),
                "control_score": scores[control_key],
                "score": scores[clip_key],
                "control_logit": logit(scores[control_key]),
                "clip_logit": logit(scores[clip_key]),
            }
            record["delta_logit"] = record["clip_logit"] - record["control_logit"]
            records.append(record)

    if not records:
        print("짝지을 클립이 없습니다.")
        return

    print(f"\n짝지어 비교한 클립 {len(records)}개 (판정 기준 {DECISION_THRESHOLD})\n")
    header = ("전환", "정답", "쌍", "logit 변화", "오른 비율", "p값", "가짜 판정 대조군→전환")
    print(f"{header[0]:<12}{header[1]:<6}{header[2]:>4}{header[3]:>14}{header[4]:>11}{header[5]:>9}{header[6]:>22}")
    for label_value, label_name in ((0, "real"), (1, "fake")):
        for kind in sorted({record["kind"] for record in records}):
            group = [record for record in records if record["label"] == label_value and record["kind"] == kind]
            if not group:
                continue
            deltas = [record["delta_logit"] for record in group]
            rose = np.mean([delta > 0 for delta in deltas])
            control_alarm = np.mean([record["control_score"] >= DECISION_THRESHOLD for record in group])
            clip_alarm = np.mean([record["score"] >= DECISION_THRESHOLD for record in group])
            print(f"{kind:<12}{label_name:<6}{len(group):>4}{np.median(deltas):>+14.3f}"
                  f"{rose:>10.0%}{wilcoxon_p(deltas):>9.4f}"
                  f"{control_alarm:>14.0%} → {clip_alarm:.0%}")

    out_path = Path(config["paths"]["results"]) / "benchmark" / "paper_effect.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\n클립별 값 저장: {out_path}")


if __name__ == "__main__":
    main()
