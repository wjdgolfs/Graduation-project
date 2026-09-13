# ============================================================================
# 데이터셋 목록(index) 만들기
# ----------------------------------------------------------------------------
# 전처리와 학습이 똑같은 영상 목록·라벨·분할을 쓰도록, 데이터셋 폴더를 읽어 표(행 목록)로 만듭니다.
#   한 행 = 영상 하나
#     video_id       : "폴더/파일이름" 형식의 고유 이름. 예) "Celeb-real/id0_0000", "Deepfakes/000_003"
#     dataset        : celebdf 또는 ffpp
#     source         : 영상이 있던 폴더(조작 기법) 이름
#     label          : 0 = real, 1 = fake   (주의: Celeb-DF 공식 목록 파일은 반대로 1 = real 입니다)
#     path           : 영상 파일 경로
#     split          : 주로 쓸 분할 (train / val / test / excluded)
#     split_official : 데이터셋이 공식으로 정한 분할
#
# Celeb-DF v2 (원 논문 재현용)
#   원 논문(이대현·문종섭 2020)은 Celeb-real 590개와 Celeb-synthesis 5,639개를 쓰고 YouTube-real 은 쓰지 않았습니다.
#   학습 real 490 / fake 5,539, 테스트 real 100 / fake 100 으로 나눴지만 어떤 영상인지는 밝히지 않았습니다.
#   그래서 split(논문 방식)은 공식 테스트 목록(List_of_testing_videos.txt)에 든 Celeb-real / Celeb-synthesis 중에서
#   real 100개, fake 100개를 고정된 난수(seed)로 뽑아 테스트로 두고, 나머지를 학습으로 둡니다. 개수는 논문과 같습니다.
#   split_official 은 공식 목록 그대로입니다(목록에 있으면 test, 없으면 train).
#
# FaceForensics++
#   원본(original)과 조작 영상 4종을 공식 분할(splits/train.json, val.json, test.json)로 나눕니다.
#   조작 영상 "A_B" 는 쌍 [A, B] 가 속한 분할을 따릅니다. 이미 받아진 파일만 목록에 넣습니다.
#
# 실행 예 (프로젝트 폴더에서): 목록만 만들고 개수를 확인
#   python preprocessing/build_index.py --dataset celebdf
# ============================================================================

import argparse
import csv
import json
import random  # 테스트 영상을 고정된 난수로 뽑습니다.
from collections import Counter
from pathlib import Path

import yaml


# __file__는 현재 파일 경로입니다. 상위 두 단계인 프로젝트 폴더를 기준으로 설정 파일을 찾습니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

INDEX_COLUMNS = ["video_id", "dataset", "source", "label", "path", "split", "split_official"]
FFPP_METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")


# 여러 스크립트가 같은 경로와 기준값을 쓰도록 YAML 설정을 딕셔너리로 읽습니다.
def load_config():
    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return yaml.safe_load(file)


# Celeb-DF v2 목록을 만듭니다. root 는 Celeb-real, Celeb-synthesis, YouTube-real 폴더가 있는 곳입니다.
def celebdf_index(root, test_per_class=100, seed=0):
    root = Path(root)

    # 공식 테스트 목록: 한 줄에 "라벨 상대경로" (예: "1 YouTube-real/00170.mp4")
    official_test = set()
    with open(root / "List_of_testing_videos.txt", "r", encoding="utf-8") as file:
        for line in file:
            parts = line.split()
            if len(parts) == 2:
                official_test.add(parts[1])

    rows = []
    for source, label in (("Celeb-real", 0), ("Celeb-synthesis", 1), ("YouTube-real", 0)):
        for path in sorted((root / source).glob("*.mp4")):
            rows.append({
                "video_id": f"{source}/{path.stem}",
                "dataset": "celebdf",
                "source": source,
                "label": label,
                "path": path.as_posix(),
                # 논문 방식 분할은 아래에서 test 를 고르고, 나머지는 train 으로 둡니다.
                "split": "excluded" if source == "YouTube-real" else "train",
                "split_official": "test" if f"{source}/{path.name}" in official_test else "train",
            })

    # random.Random(seed): 전역 난수와 분리된 난수 생성기. 같은 seed 면 언제 실행해도 같은 영상이 뽑힙니다.
    rng = random.Random(seed)
    for source in ("Celeb-real", "Celeb-synthesis"):
        in_official = [row for row in rows if row["source"] == source and row["split_official"] == "test"]
        others = [row for row in rows if row["source"] == source and row["split_official"] == "train"]
        rng.shuffle(in_official)
        chosen = in_official[:test_per_class]
        # 공식 목록에 그 폴더 영상이 모자라면 나머지 영상에서 채웁니다.
        if len(chosen) < test_per_class:
            rng.shuffle(others)
            chosen += others[:test_per_class - len(chosen)]
        for row in chosen:
            row["split"] = "test"

    return rows


# FaceForensics++ 목록을 만듭니다. root 는 original_sequences, manipulated_sequences, splits 폴더가 있는 곳입니다.
def ffpp_index(root, compression="c23", methods=FFPP_METHODS):
    root = Path(root)

    # {영상 번호 또는 "A_B": 분할 이름} 표를 만듭니다.
    split_of = {}
    for split in ("train", "val", "test"):
        with open(root / "splits" / f"{split}.json", "r", encoding="utf-8") as file:
            for first, second in json.load(file):
                for name in (first, second, f"{first}_{second}", f"{second}_{first}"):
                    split_of[name] = split

    rows = []
    original_dir = root / "original_sequences" / "youtube" / compression / "videos"
    folders = [("original", original_dir, 0)] + [
        (method, root / "manipulated_sequences" / method / compression / "videos", 1)
        for method in methods
    ]
    for source, folder, label in folders:
        # 받는 중인 파일은 "*.mp4.part" 라서 "*.mp4" 로 찾으면 빠집니다.
        for path in sorted(folder.glob("*.mp4")):
            split = split_of.get(path.stem, "unknown")
            rows.append({
                "video_id": f"{source}/{path.stem}",
                "dataset": "ffpp",
                "source": source,
                "label": label,
                "path": path.as_posix(),
                "split": split,
                "split_official": split,
            })

    return rows


# 목록을 CSV 로 저장합니다.
def write_index(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# 설정 파일의 경로로 데이터셋 목록을 만듭니다.
def build_rows(dataset, config):
    raw = Path(config["paths"]["raw"])
    settings = config["preprocess"]
    if dataset == "celebdf":
        return celebdf_index(
            raw / "Celeb-DF-v2",
            test_per_class=settings["celebdf_test_per_class"],
            seed=settings["split_seed"],
        )
    if dataset == "ffpp":
        return ffpp_index(raw / "FaceForensics++")
    raise ValueError(f"알 수 없는 데이터셋입니다: {dataset}")


# (분할, 출처, 라벨) 조합마다 영상 수를 세어 보여줍니다.
def print_counts(rows):
    for column in ("split", "split_official"):
        counts = Counter((row[column], row["source"]) for row in rows)
        print(f"[{column}]")
        for (split, source), count in sorted(counts.items()):
            print(f"  {split:<9} {source:<16} {count:>5}개")


def main():
    parser = argparse.ArgumentParser(description="데이터셋 영상 목록과 분할을 만듭니다.")
    parser.add_argument("--dataset", required=True, choices=["celebdf", "ffpp"])
    args = parser.parse_args()

    config = load_config()
    rows = build_rows(args.dataset, config)
    out_path = Path(config["preprocess"]["output_dir"]) / args.dataset / "index.csv"
    write_index(rows, out_path)

    print(f"영상 {len(rows)}개 → {out_path}")
    print_counts(rows)


# 이 파일을 직접 실행할 때만 main() 을 부릅니다.
if __name__ == "__main__":
    main()
