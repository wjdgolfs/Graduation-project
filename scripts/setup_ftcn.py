# ============================================================================
# FTCN 준비: 공식 코드와 가중치를 "정해진 버전"으로 받고 검증하기
# ----------------------------------------------------------------------------
# 왜 필요한가
#   FTCN 저장소에는 라이선스가 없어서 코드를 이 저장소(git)에 넣지 않았습니다(third_party/ 는 git 에서 제외).
#   대신 이 스크립트가 "어느 버전을 받았는지"를 기록해 두고, 다른 컴퓨터에서도 똑같은 버전을 다시 받게 합니다.
#   실험 결과를 나중에 재현하려면 코드와 가중치가 한 글자도 다르지 않아야 하기 때문입니다.
#
# 확인하는 두 가지
#   1) 코드: git 커밋 번호(해시)가 기록한 값과 같은지
#      커밋 번호는 그 시점 코드 전체를 대표하는 고유한 값이라, 같으면 코드도 같습니다.
#   2) 가중치: 파일의 sha256 값이 기록한 값과 같은지
#      sha256 은 파일 내용으로 계산한 64자리 "지문"입니다. 1바이트만 달라도 완전히 다른 값이 나옵니다.
#      다운로드가 중간에 끊겨 파일이 망가졌거나, 다른 파일로 바뀌었으면 여기서 걸립니다.
#
# 출력: config.yaml 의 ftcn.repo_dir (코드), ftcn.weights_dir (가중치 3개)
#
# 실행 예 (프로젝트 폴더에서)
#   python scripts/setup_ftcn.py          없으면 받고, 있으면 검증만
#   python scripts/setup_ftcn.py --check  받지 않고 검증만
# ============================================================================

import argparse
import hashlib  # sha256 을 계산합니다.
import subprocess  # git 명령을 실행합니다.
import urllib.request  # 인터넷에서 파일을 받습니다.
from pathlib import Path

import yaml


# __file__는 현재 파일 경로입니다. 상위 두 단계인 프로젝트 폴더를 기준으로 설정 파일을 찾습니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

REPO_URL = "https://github.com/yinglinzheng/FTCN.git"
# 2026-09-12 에 받은 커밋입니다. 다른 커밋이면 결과 재현이 보장되지 않습니다.
REPO_COMMIT = "853256a888bd5c4879955c06a4c5a666740f91a4"

# (파일 이름, 받는 주소, sha256)
#   ftcn_tt.pth                                   : FTCN 본체
#   mobilenet0.25_Final.pth                       : 얼굴 검출기(RetinaFace)
#   mobilenet_224_model_best_gdconv_external.pth  : 68점 랜드마크
WEIGHTS = [
    (
        "ftcn_tt.pth",
        "https://github.com/yinglinzheng/FTCN/releases/download/weights/ftcn_tt.pth",
        "3d08fc78174bbfdb3fbede317b95dcac9095202ed7113649eecf6e2ef5ee77d8",
    ),
    (
        "mobilenet0.25_Final.pth",
        "https://github.com/yinglinzheng/face_weights/releases/download/v1/mobilenet0.25_Final.pth",
        "2979b33ffafda5d74b6948cd7a5b9a7a62f62b949cef24e95fd15d2883a65220",
    ),
    (
        "mobilenet_224_model_best_gdconv_external.pth",
        "https://github.com/yinglinzheng/face_weights/releases/download/v1/mobilenet_224_model_best_gdconv_external.pth",
        "826b3c902e70e1eeb177f35c73198af0714f74502fe7bd3cdea42e847b1ca30f",
    ),
]


# 여러 스크립트가 같은 경로와 기준값을 쓰도록 YAML 설정을 딕셔너리로 읽습니다.
def load_config():
    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return yaml.safe_load(file)


# 파일의 sha256 을 계산합니다. 60MB 파일도 메모리를 적게 쓰도록 1MB 씩 나눠 읽습니다.
#   iter(함수, b"") : 함수가 빈 바이트(파일 끝)를 돌려줄 때까지 계속 불러 주는 반복자입니다.
def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    # hexdigest(): 계산 결과를 0-9, a-f 로 된 64자리 문자열로 돌려줍니다.
    return digest.hexdigest()


# 코드가 없으면 받아서 정해진 커밋으로 맞춥니다. 있으면 커밋이 맞는지만 확인합니다.
# 반환값: 준비가 됐으면 True
def ensure_repo(repo_dir, check_only):
    # .git 폴더가 있으면 git 저장소가 이미 받아져 있다는 뜻입니다.
    if not (repo_dir / ".git").exists():
        if check_only:
            print(f"[코드] 없음: {repo_dir}")
            return False
        print(f"[코드] 받는 중: {REPO_URL}")
        # check=True: 명령이 실패하면 파이썬 오류를 내서 멈춥니다.
        subprocess.run(["git", "clone", "--quiet", REPO_URL, str(repo_dir)], check=True)
        # 받은 뒤 기록한 커밋으로 되돌려 맞춥니다(저장소가 나중에 업데이트돼도 같은 버전을 쓰기 위해).
        subprocess.run(["git", "-C", str(repo_dir), "checkout", "--quiet", REPO_COMMIT], check=True)

    # "git rev-parse HEAD" 는 지금 받아져 있는 코드의 커밋 번호를 출력합니다.
    #   capture_output=True, text=True : 출력을 화면에 찍지 않고 문자열로 받습니다.
    head = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    if head != REPO_COMMIT:
        print(f"[코드] 커밋이 다릅니다: {head} (기대값 {REPO_COMMIT})")
        return False

    print(f"[코드] 확인: {repo_dir} @ {head[:10]}")
    return True


# 가중치가 없으면 받고, 모든 파일의 sha256 을 확인합니다.
# 반환값: 세 파일 모두 준비됐으면 True
def ensure_weights(weights_dir, check_only):
    weights_dir.mkdir(parents=True, exist_ok=True)
    all_ok = True

    for name, url, expected in WEIGHTS:
        path = weights_dir / name

        if not path.exists():
            if check_only:
                print(f"[가중치] 없음: {name}")
                all_ok = False
                continue
            print(f"[가중치] 받는 중: {name}")
            # 먼저 ".part" 이름으로 받고, 다 받은 뒤에 진짜 이름으로 바꿉니다.
            # 받는 도중에 끊기면 .part 파일만 남아서, 망가진 파일을 완성본으로 착각하지 않습니다.
            part = path.with_name(name + ".part")
            urllib.request.urlretrieve(url, part)
            part.replace(path)

        actual = sha256_of(path)
        if actual != expected:
            print(f"[가중치] 해시 불일치: {name} ({actual[:12]} / 기대값 {expected[:12]})")
            all_ok = False
        else:
            print(f"[가중치] 확인: {name}")

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="FTCN 코드와 가중치를 준비하고 검증합니다."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="받지 않고 검사만 합니다."
    )
    args = parser.parse_args()

    config = load_config()
    repo_dir = PROJECT_ROOT / config["ftcn"]["repo_dir"]
    weights_dir = Path(config["ftcn"]["weights_dir"])

    repo_ok = ensure_repo(repo_dir, args.check)
    weights_ok = ensure_weights(weights_dir, args.check)

    # SystemExit 에 문자열을 주면 그 문장을 출력하고 "실패(종료 코드 1)"로 끝납니다.
    if not (repo_ok and weights_ok):
        raise SystemExit("FTCN 준비가 끝나지 않았습니다. 위 메시지를 확인하세요.")

    print("FTCN 준비 완료")


# 이 파일을 직접 실행할 때만 main() 을 부릅니다.
if __name__ == "__main__":
    main()
