# ============================================================================
# FaceForensics++ 병렬 다운로드
# ----------------------------------------------------------------------------
# 무엇을 하나
#   FF++ 원본 영상, 조작 영상 4종, 조작 마스크 4종을 우선순위 순서대로, 여러 연결을 동시에 써서 내려받습니다.
#   공식 스크립트(scripts/download_faceforensics.py)와 같은 서버, 같은 폴더 구조를 쓰므로 둘을 번갈아 써도 이어받기가 됩니다.
#
# 입력: 없음 / 출력: config.yaml 의 raw 경로 아래 FaceForensics++ 폴더
#
# 실행 예 (프로젝트 폴더에서)
#   python scripts/download_ffpp_batch.py --list                 받을 목록과 예상 용량만 보기
#   python scripts/download_ffpp_batch.py --workers 3            전체를 순서대로 받기
#   python scripts/download_ffpp_batch.py --only original        원본 영상만 받기
#
# 왜 병렬로 받는가
#   연결 하나로는 약 24 KB/s 밖에 안 나오지만 4개를 동시에 쓰면 약 161 KB/s 가 나옵니다(2026-09-06 실측).
#   서버가 연결 하나당 속도를 제한하기 때문입니다. 다만 8개로 늘리면 서버가 응답을 끊었으므로
#   동시 연결 수는 4개를 넘기지 않습니다.
#
# 스레드(thread)란
#   한 프로그램 안에서 여러 일을 동시에 진행하는 실행 흐름입니다.
#   파일 받기는 대부분의 시간을 "서버 응답 기다리기"에 쓰기 때문에, 스레드 여러 개가 동시에 기다리게 하면 빨라집니다.
#
# 이어받기
#   이미 받은 파일은 건너뛰고, 받는 중인 파일은 ".part" 이름으로 저장했다가 다 받으면 진짜 이름으로 바꿉니다.
#   그래서 중간에 끊겨도 다시 실행하면 남은 것만 받고, 반쯤 받은 파일을 완성본으로 착각하지 않습니다.
# ============================================================================

import argparse
import json  # 서버의 파일 목록(filelist.json)을 읽습니다.
import shutil  # 디스크 여유 공간을 확인합니다.
import time  # 실패했을 때 잠시 기다립니다.
import urllib.error
import urllib.request  # 인터넷에서 파일을 받습니다.
from concurrent.futures import ThreadPoolExecutor  # 스레드 여러 개로 일을 나눠 맡깁니다.
from pathlib import Path

import yaml
from tqdm import tqdm  # 진행률 막대를 표시합니다.


# __file__는 현재 파일 경로입니다. 상위 두 단계인 프로젝트 폴더를 기준으로 설정 파일을 찾습니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# 공식 스크립트의 EU2 서버입니다. http로 접속하면 301로 https에 넘겨지므로 처음부터 https를 씁니다.
# EU(canis.vc.in.tum.de:8100)와 CA(falas.cmpt.sfu.ca:8100)는 2026-09-06 기준 연결되지 않습니다.
BASE_URL = "https://kaldir.vc.in.tum.de/faceforensics/v3/"
FILELIST_URL = BASE_URL + "misc/filelist.json"
TOS_URL = "https://kaldir.vc.in.tum.de/faceforensics/webpage/FaceForensics_TOS.pdf"

# 서버가 응답을 끊기 시작하는 지점이 8개였으므로 그 절반을 상한으로 둡니다.
MAX_WORKERS = 4

# 데이터 종류별 서버 경로. 공식 스크립트의 DATASETS와 같습니다.
DATASET_PATHS = {
    "original": "original_sequences/youtube",
    "Deepfakes": "manipulated_sequences/Deepfakes",
    "Face2Face": "manipulated_sequences/Face2Face",
    "FaceSwap": "manipulated_sequences/FaceSwap",
    "NeuralTextures": "manipulated_sequences/NeuralTextures",
}

# 받을 작업 목록. 위에 있을수록 먼저 필요한 데이터라서 먼저 받습니다. 예상 용량은 공식 문서 기준의 어림값입니다.
#   name    : 작업 이름(--only, --skip 에 씁니다)
#   dataset : DATASET_PATHS 의 키
#   type    : videos(영상) 또는 masks(조작된 픽셀 위치를 흰색으로 표시한 영상)
JOBS = [
    {
        "name": "original",
        "dataset": "original",
        "type": "videos",
        "size_gb": 2.0,
        "reason": "real 영상 1000개. 0단계 파일럿은 이것만 있어도 시작할 수 있습니다.",
    },
    {
        "name": "Deepfakes",
        "dataset": "Deepfakes",
        "type": "videos",
        "size_gb": 2.0,
        "reason": "조작 영상. 학습과 탐지 성능 측정에 필요합니다.",
    },
    {
        "name": "Face2Face",
        "dataset": "Face2Face",
        "type": "videos",
        "size_gb": 2.0,
        "reason": "조작 영상.",
    },
    {
        "name": "FaceSwap",
        "dataset": "FaceSwap",
        "type": "videos",
        "size_gb": 2.0,
        "reason": "조작 영상. 경계 블렌딩 아티팩트가 가장 뚜렷한 기법입니다.",
    },
    {
        "name": "NeuralTextures",
        "dataset": "NeuralTextures",
        "type": "videos",
        "size_gb": 2.0,
        "reason": "조작 영상. 입 주변만 바꾸므로 부위별 분석의 대조군이 됩니다.",
    },
    {
        "name": "masks_Deepfakes",
        "dataset": "Deepfakes",
        "type": "masks",
        "size_gb": 1.5,
        "reason": "조작 픽셀 위치. 부위별 기여도 분석의 정답지입니다.",
    },
    {
        "name": "masks_Face2Face",
        "dataset": "Face2Face",
        "type": "masks",
        "size_gb": 1.5,
        "reason": "조작 픽셀 위치.",
    },
    {
        "name": "masks_FaceSwap",
        "dataset": "FaceSwap",
        "type": "masks",
        "size_gb": 1.5,
        "reason": "조작 픽셀 위치.",
    },
    {
        "name": "masks_NeuralTextures",
        "dataset": "NeuralTextures",
        "type": "masks",
        "size_gb": 1.5,
        "reason": "조작 픽셀 위치.",
    },
]


# 여러 스크립트가 같은 경로를 쓰도록 YAML 설정을 딕셔너리로 읽습니다.
def load_config():
    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return yaml.safe_load(file)


# 지정한 경로가 속한 드라이브의 남은 공간을 GB 단위로 돌려줍니다.
def free_space_gb(path):
    # 아직 없는 폴더면 존재하는 상위 폴더를 기준으로 재야 합니다.
    # probe.parent 는 한 단계 위 폴더입니다. 드라이브 맨 위(D:\)에 오면 parent 가 자기 자신이라 멈춥니다.
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    # 1024 ** 3 = 1GB 의 바이트 수
    return shutil.disk_usage(probe).free / 1024 ** 3


# 서버에서 500개 쌍 목록을 받아 옵니다. 원본과 조작 영상의 파일 이름이 여기서 나옵니다.
#   목록 모양: [["585", "599"], ["469", "481"], ...]
def fetch_file_pairs():
    with urllib.request.urlopen(FILELIST_URL, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


# 공식 스크립트와 같은 규칙으로 받을 파일 이름을 만듭니다.
#   original : 쌍을 펼쳐 1000개  → "585.mp4", "599.mp4", ...
#   조작 영상 : 쌍의 양방향으로 1000개 → "585_599.mp4", "599_585.mp4", ...
#              "A_B" 는 A 영상에 B 의 얼굴(또는 표정)을 입힌 조작 영상입니다.
def build_filelist(dataset, file_pairs):
    if dataset == "original":
        names = [name for pair in file_pairs for name in pair]
    else:
        names = []
        for pair in file_pairs:
            names.append("_".join(pair))
            # pair[::-1] 은 쌍의 순서를 뒤집은 것입니다. ["585", "599"] → ["599", "585"]
            names.append("_".join(pair[::-1]))

    return [name + ".mp4" for name in names]


# 작업 하나가 쓸 서버 주소와 저장 폴더를 정합니다. masks는 압축률 폴더 대신 masks 폴더를 씁니다.
#   videos 예: .../manipulated_sequences/Deepfakes/c23/videos/
#   masks  예: .../manipulated_sequences/Deepfakes/masks/videos/
def resolve_urls(job, output_root, compression):
    dataset_path = DATASET_PATHS[job["dataset"]]

    if job["type"] == "masks":
        url = BASE_URL + dataset_path + "/masks/videos/"
        out_dir = Path(output_root) / dataset_path / "masks" / "videos"
    else:
        url = BASE_URL + dataset_path + "/" + compression + "/videos/"
        out_dir = Path(output_root) / dataset_path / compression / "videos"

    return url, out_dir


# 파일 하나를 받습니다. 이미 있으면 건너뛰고, 실패하면 잠시 쉬었다 다시 시도합니다.
# 받는 도중에는 .part 이름을 쓰다가 다 받은 뒤 이름을 바꿔, 중단된 파일이 완성본으로 오인되지 않게 합니다.
# 반환값: "ok"(받음), "skip"(이미 있음), "fail"(여러 번 시도해도 실패)
def download_one(url, out_path, retries=3):
    if out_path.exists():
        return "skip"

    # with_suffix: 확장자를 바꿉니다. "585.mp4" → "585.mp4.part"
    part_path = out_path.with_suffix(out_path.suffix + ".part")

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
            part_path.write_bytes(data)
            # replace: 이름 바꾸기. 같은 이름이 있어도 덮어씁니다.
            part_path.replace(out_path)
            return "ok"
        except (urllib.error.URLError, OSError, TimeoutError):
            # 서버가 연결을 끊는 경우가 있어 뒤로 갈수록 더 오래 기다립니다. (5초, 10초)
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))

    if part_path.exists():
        part_path.unlink()

    return "fail"


# 파일 목록을 여러 연결로 나눠 받습니다. 반환값은 (개수 요약, 실패한 파일 목록)입니다.
def download_parallel(filelist, base_url, out_dir, workers):
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {"ok": 0, "skip": 0, "fail": 0}
    failed = []

    # 스레드 하나가 맡을 일: 파일 이름을 받아 다운로드하고 (이름, 결과) 를 돌려줍니다.
    def work(filename):
        return filename, download_one(base_url + filename, out_dir / filename)

    # ThreadPoolExecutor: 스레드 workers 개를 만들어 일을 나눠 줍니다.
    # pool.map(work, filelist) 는 목록의 각 항목에 work 를 동시에 실행하고, 결과는 목록 순서대로 돌려줍니다.
    # tqdm(...) 으로 감싸면 결과가 하나 나올 때마다 진행률 막대가 올라갑니다.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for filename, result in tqdm(
            pool.map(work, filelist),
            total=len(filelist),
            unit="개"
        ):
            counts[result] += 1
            if result == "fail":
                failed.append(filename)

    return counts, failed


def main():
    parser = argparse.ArgumentParser(
        description="FaceForensics++ 데이터를 우선순위 순서대로 내려받습니다."
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="지정한 작업 이름만 실행합니다. 이름은 --list로 확인하세요."
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        help="지정한 작업 이름을 건너뜁니다."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="작업 목록과 예상 용량만 출력하고 끝냅니다."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="동시 연결 수. 서버가 막기 시작하는 지점이 8이므로 4개로 제한됩니다."
    )
    parser.add_argument(
        "--compression",
        default="c23",
        choices=["raw", "c23", "c40"],
        help="raw는 500GB이므로 쓰지 마세요."
    )
    parser.add_argument(
        "--reserve-gb",
        type=float,
        default=20.0,
        help="이만큼의 여유 공간은 남겨둡니다. 전처리 결과와 전환 합성본이 들어갈 자리입니다."
    )
    args = parser.parse_args()

    # 사용자가 큰 값을 줘도 1 ~ MAX_WORKERS 사이로 제한합니다.
    workers = max(1, min(args.workers, MAX_WORKERS))

    config = load_config()
    output_root = Path(config["paths"]["raw"]) / "FaceForensics++"

    # 실행할 작업을 고릅니다. --only를 줘도 순서는 JOBS에 적힌 우선순위를 따릅니다.
    jobs = JOBS
    if args.only is not None:
        jobs = [job for job in jobs if job["name"] in set(args.only)]
    jobs = [job for job in jobs if job["name"] not in set(args.skip)]

    # --list: 목록만 보여주고 끝냅니다.
    if args.list:
        print("출력 위치:", output_root)
        print(f"현재 여유 공간: {free_space_gb(output_root):.1f} GB")
        print("-" * 74)
        for job in JOBS:
            # {값:24s} 는 24칸 너비로 왼쪽 정렬, {값:5.1f} 는 5칸 너비에 소수점 한 자리입니다.
            print(f"{job['name']:24s} 약 {job['size_gb']:5.1f} GB  {job['reason']}")
        print("-" * 74)
        print(f"전체 예상 용량: 약 {sum(job['size_gb'] for job in JOBS):.1f} GB")
        print()
        print("유튜브 원본(38.5GB, 단일 zip)은 여기서 다루지 않습니다. 공식 스크립트로 받으세요:")
        print("  python scripts/download_faceforensics.py <경로> -d original_youtube_videos --server EU2")
        return

    print("출력 위치:", output_root)
    print(f"현재 여유 공간: {free_space_gb(output_root):.1f} GB")
    print(f"동시 연결 수: {workers}")
    print(f"작업 {len(jobs)}개, 예상 용량 약 {sum(job['size_gb'] for job in jobs):.1f} GB")
    print()

    # 약관 동의는 사람이 직접 밝혀야 합니다. 이 코드가 대신 판단하지 않습니다.
    print("FaceForensics 이용약관:", TOS_URL)
    answer = input("위 약관에 동의하셨습니까? 동의하셨다면 y를 입력하세요: ").strip().lower()
    if answer != "y":
        print("중단합니다.")
        return
    print()

    print("파일 목록을 받는 중...")
    file_pairs = fetch_file_pairs()
    print(f"쌍 {len(file_pairs)}개 확인")
    print()

    for job in jobs:
        # 작업마다 여유 공간을 다시 확인합니다. 앞 작업에서 용량을 썼기 때문입니다.
        free_gb = free_space_gb(output_root)
        needed_gb = job["size_gb"] + args.reserve_gb

        print("=" * 74)
        print(f"[{job['name']}] 예상 {job['size_gb']:.1f} GB / 여유 {free_gb:.1f} GB")
        print("  용도:", job["reason"])

        if free_gb < needed_gb:
            print(
                f"  건너뜁니다. 여유 공간이 부족합니다 "
                f"(예상 {job['size_gb']:.1f} GB + 예비 {args.reserve_gb:.1f} GB 필요)."
            )
            continue

        base_url, out_dir = resolve_urls(job, output_root, args.compression)
        filelist = build_filelist(job["dataset"], file_pairs)

        print("  주소:", base_url)
        print("  저장:", out_dir)

        counts, failed = download_parallel(filelist, base_url, out_dir, workers)

        print(
            f"  받음 {counts['ok']} / 이미 있음 {counts['skip']} / 실패 {counts['fail']}"
        )

        if failed:
            # 실패가 몰리면 서버가 막고 있는 것이므로 다음 작업으로 넘어가지 않습니다.
            print("  실패한 파일 예:", failed[:5])
            if counts["fail"] > counts["ok"]:
                print("  실패가 너무 많습니다. 서버 제한일 수 있으니 15분쯤 뒤에 다시 실행하세요.")
                return
            print("  다시 실행하면 실패한 것만 이어서 받습니다.")

    print("=" * 74)
    print("요청한 작업을 모두 마쳤습니다.")


# 이 파일을 직접 실행할 때만 main() 을 부릅니다.
if __name__ == "__main__":
    main()
