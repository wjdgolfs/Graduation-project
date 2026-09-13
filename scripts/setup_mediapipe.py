# ============================================================================
# MediaPipe 준비: 얼굴 랜드마크(478점) 모델 파일을 받고 검증하기
# ----------------------------------------------------------------------------
# 왜 필요한가
#   MediaPipe 의 FaceLandmarker 는 모델 파일(.task)이 따로 있어야 동작합니다.
#   이 스크립트가 받는 주소와 sha256 을 기록해 두어, 다른 컴퓨터에서도 같은 모델로 전처리를 재현하게 합니다.
#   (sha256 이 무엇인지는 scripts/setup_ftcn.py 머리말에 설명해 두었습니다.)
#
# 확인하는 것
#   1) mediapipe 패키지가 import 되는지, Tasks API 의 FaceLandmarker 가 있는지
#   2) 모델 파일의 sha256 이 기록한 값과 같은지
#
# mediapipe 설치 (requirements.txt 와 별도로 한 번만)
#   mediapipe 는 opencv-contrib-python 을 요구해서 그대로 설치하면 opencv-python 과 cv2 가 이중 설치됩니다.
#   그래서 의존성 자동 설치를 끄고 설치합니다(빠진 의존성 flatbuffers, sounddevice 는 requirements.txt 에 있음):
#     .venv/Scripts/python.exe -m pip install --no-deps mediapipe==1.0.1
#
# 출력: config.yaml 의 mediapipe.face_landmarker 경로에 모델 파일
#
# 실행 예 (프로젝트 폴더에서)
#   python scripts/setup_mediapipe.py          없으면 받고, 있으면 검증만
#   python scripts/setup_mediapipe.py --check  받지 않고 검증만
# ============================================================================

import argparse
import hashlib  # sha256 을 계산합니다.
import urllib.request  # 인터넷에서 파일을 받습니다.
from pathlib import Path

import yaml


# __file__는 현재 파일 경로입니다. 상위 두 단계인 프로젝트 폴더를 기준으로 설정 파일을 찾습니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# 버전 1 모델입니다. "latest" 주소와 같은 파일이지만(2026-09-13 확인), latest 는 나중에 바뀔 수 있어서 버전 번호로 고정합니다.
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
# 동작을 확인한 mediapipe 버전. 다른 버전이면 경고만 합니다.
MEDIAPIPE_VERSION = "1.0.1"


# 여러 스크립트가 같은 경로와 기준값을 쓰도록 YAML 설정을 딕셔너리로 읽습니다.
def load_config():
    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return yaml.safe_load(file)


# 파일의 sha256 을 계산합니다. 1MB 씩 나눠 읽어 메모리를 적게 씁니다.
def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# mediapipe 가 설치돼 있고 FaceLandmarker 를 쓸 수 있는지 확인합니다. 반환값: 쓸 수 있으면 True
def check_package():
    # import 를 함수 안에서 하는 이유: 설치가 안 돼 있어도 스크립트가 멈추지 않고 안내 문구를 보여주기 위해서입니다.
    try:
        import mediapipe
        from mediapipe.tasks.python import vision
    except ImportError as error:
        print(f"[패키지] mediapipe 를 불러오지 못했습니다: {error}")
        print("         .venv/Scripts/python.exe -m pip install --no-deps mediapipe==1.0.1")
        return False

    # mediapipe 1.0 부터 예전 solutions.face_mesh 는 없고 Tasks API 의 FaceLandmarker 만 있습니다.
    if not hasattr(vision, "FaceLandmarker"):
        print("[패키지] 이 mediapipe 버전에는 FaceLandmarker 가 없습니다.")
        return False

    note = "" if mediapipe.__version__ == MEDIAPIPE_VERSION else f" (동작 확인한 버전 {MEDIAPIPE_VERSION} 과 다름)"
    print(f"[패키지] 확인: mediapipe {mediapipe.__version__}{note}")
    return True


# 모델 파일이 없으면 받고, sha256 을 확인합니다. 반환값: 준비됐으면 True
def ensure_model(model_path, check_only):
    if not model_path.exists():
        if check_only:
            print(f"[모델] 없음: {model_path}")
            return False
        print(f"[모델] 받는 중: {MODEL_URL}")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        # ".part" 이름으로 받은 뒤 다 받으면 진짜 이름으로 바꿉니다(끊긴 파일을 완성본으로 착각하지 않게).
        part = model_path.with_name(model_path.name + ".part")
        urllib.request.urlretrieve(MODEL_URL, part)
        part.replace(model_path)

    actual = sha256_of(model_path)
    if actual != MODEL_SHA256:
        print(f"[모델] 해시 불일치: {actual[:12]} / 기대값 {MODEL_SHA256[:12]}")
        return False

    print(f"[모델] 확인: {model_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="MediaPipe 얼굴 랜드마크 모델을 준비하고 검증합니다."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="받지 않고 검사만 합니다."
    )
    args = parser.parse_args()

    config = load_config()
    model_path = Path(config["mediapipe"]["face_landmarker"])

    package_ok = check_package()
    model_ok = ensure_model(model_path, args.check)

    if not (package_ok and model_ok):
        raise SystemExit("MediaPipe 준비가 끝나지 않았습니다. 위 메시지를 확인하세요.")

    print("MediaPipe 준비 완료")


# 이 파일을 직접 실행할 때만 main() 을 부릅니다.
if __name__ == "__main__":
    main()
