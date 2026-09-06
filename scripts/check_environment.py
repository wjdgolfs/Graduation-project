# 영상 처리와 텐서 실험에 필요한 라이브러리 및 GPU 상태를 확인합니다.
# PyTorch/OpenCV 버전, CUDA 사용 가능 여부, 데이터 폴더 존재 여부를 출력합니다.
# 폴더를 만들거나 데이터를 수정하지 않습니다. import가 실패하면 해당 환경의 설치 상태를 확인하세요.
# 이 파일은 함수로 감싸지 않아 직접 실행할 때뿐 아니라 import할 때도 점검 코드가 실행됩니다.

from pathlib import Path

import cv2
import torch
import yaml


# __file__는 현재 파일 경로입니다. 상위 두 단계인 프로젝트 폴더를 기준으로 설정 파일을 찾습니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


print("Python environment test")
print("-" * 50)

print("PyTorch:", torch.__version__)
# CUDA는 PyTorch가 지원되는 NVIDIA GPU를 계산에 사용할 때 쓰는 기능입니다.
print("CUDA available:", torch.cuda.is_available())

# GPU를 쓸 수 있을 때만 첫 번째 GPU(번호 0)의 이름과 메모리를 조회합니다.
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(
            # 사용 가능한 잔여량이 아니라 장치의 전체 메모리 용량입니다. 1024³으로 나누므로 정확한 단위는 GiB입니다.
            torch.cuda.get_device_properties(0).total_memory
            / 1024 ** 3,
            2
        ),
        "GB"
    )

print("OpenCV:", cv2.__version__)

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as file:
    config = yaml.safe_load(file)

# 설정에 적힌 데이터 폴더의 존재만 확인합니다. 하위 데이터 파일이 준비됐는지는 검사하지 않습니다.
data_root = Path(config["paths"]["data_root"])

print("Data root:", data_root)
print("Data root exists:", data_root.exists())
