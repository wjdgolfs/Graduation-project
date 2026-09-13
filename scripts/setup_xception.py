# ============================================================================
# XceptionNet 준비: 원 논문 재현용 ImageNet 사전학습 가중치를 받고 검증하기
# ----------------------------------------------------------------------------
# 왜 필요한가
#   원 논문 모델(models/paper_bclstm.py)은 프레임마다 XceptionNet 으로 특징을 뽑습니다.
#   Celeb-DF 학습 영상(real 490개)만으로 XceptionNet 을 처음부터 학습하기는 어려워서, ImageNet 사전학습 가중치에서 시작합니다.
#   받는 파일은 Keras 의 ImageNet Xception 가중치를 PyTorch 로 옮긴 것으로, timm 의 legacy_xception 이 기본으로 쓰는 파일입니다.
#   timm 이 알아서 받게 두면 사용자 폴더의 캐시에 들어가서, 다른 가중치들과 같이 checkpoints 폴더에 두려고 따로 받습니다.
#
# 확인하는 것
#   1) 파일의 sha256 이 파일 이름에 적힌 앞자리("43020ad28")로 시작하는지
#      (PyTorch Hub 의 파일 이름 규칙: 이름-<sha256 앞자리>.pth. sha256 은 scripts/setup_ftcn.py 머리말 참고)
#   2) 모델 코드와 같은 방법으로 timm legacy_xception 에 가중치가 들어가는지
#
# 출력: config.yaml 의 paper_model.backbone_weights 경로에 가중치 파일 (약 88MB)
#
# 실행 예 (프로젝트 폴더에서)
#   python scripts/setup_xception.py          없으면 받고, 있으면 검증만
#   python scripts/setup_xception.py --check  받지 않고 검증만
# ============================================================================

import argparse
import hashlib  # sha256 을 계산합니다.
import urllib.request  # 인터넷에서 파일을 받습니다.
from pathlib import Path

import yaml


# __file__는 현재 파일 경로입니다. 상위 두 단계인 프로젝트 폴더를 기준으로 설정 파일을 찾습니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# timm 1.0.29 의 legacy_xception 기본 가중치 주소(pretrained_cfg 의 url)입니다.
WEIGHTS_URL = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/xception-43020ad28.pth"
SHA256_PREFIX = "43020ad28"


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


# 가중치 파일이 없으면 받고, sha256 앞자리를 확인합니다. 반환값: 준비됐으면 True
def ensure_weights(weights_path, check_only):
    if not weights_path.exists():
        if check_only:
            print(f"[가중치] 없음: {weights_path}")
            return False
        print(f"[가중치] 받는 중: {WEIGHTS_URL}")
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        # ".part" 이름으로 받은 뒤 다 받으면 진짜 이름으로 바꿉니다(끊긴 파일을 완성본으로 착각하지 않게).
        part = weights_path.with_name(weights_path.name + ".part")
        urllib.request.urlretrieve(WEIGHTS_URL, part)
        part.replace(weights_path)

    actual = sha256_of(weights_path)
    if not actual.startswith(SHA256_PREFIX):
        print(f"[가중치] 해시 불일치: {actual[:12]} / 기대 앞자리 {SHA256_PREFIX}")
        return False

    print(f"[가중치] 확인: {weights_path} (sha256 {actual})")
    return True


# 모델 코드(models/paper_bclstm.py)와 같은 방법으로 불러 봅니다. 반환값: 제대로 들어갔으면 True
# 분류층(fc)을 뺀 가중치 중 하나라도 이름이나 모양이 맞지 않으면 timm 이 오류를 냅니다.
def check_loading(weights_path):
    import timm
    import torch

    try:
        model = timm.create_model(
            "legacy_xception",
            pretrained=True,
            pretrained_cfg_overlay=dict(file=str(weights_path)),
            num_classes=0,
            global_pool="",
        )
    except Exception as error:
        print(f"[불러오기] 실패: {error}")
        return False

    # 무작위 초기값이 아니라 파일의 값이 들어갔는지 첫 합성곱 가중치로 확인합니다.
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    if not torch.equal(model.conv1.weight, state["conv1.weight"]):
        print("[불러오기] 첫 합성곱 가중치가 파일 값과 다릅니다.")
        return False

    print(f"[불러오기] 확인: 파일의 가중치 {len(state)}개 중 분류층을 뺀 나머지가 모델에 들어갔습니다.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="원 논문 재현용 XceptionNet ImageNet 가중치를 준비하고 검증합니다."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="받지 않고 검사만 합니다."
    )
    args = parser.parse_args()

    config = load_config()
    weights_path = Path(config["paper_model"]["backbone_weights"])

    weights_ok = ensure_weights(weights_path, args.check)
    loading_ok = weights_ok and check_loading(weights_path)

    if not loading_ok:
        raise SystemExit("XceptionNet 준비가 끝나지 않았습니다. 위 메시지를 확인하세요.")

    print("XceptionNet 준비 완료")


# 이 파일을 직접 실행할 때만 main() 을 부릅니다.
if __name__ == "__main__":
    main()
