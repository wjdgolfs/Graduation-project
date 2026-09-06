# 영상과 장면 전환 정보를 어떤 배열 모양으로 다룰지 확인하는 작은 실행 예제입니다.
# 텐서(tensor)는 숫자를 여러 축으로 배열한 자료구조로, PyTorch가 계산에 사용합니다.
# 실제 영상이나 CSV를 읽지 않고 무작위 영상 텐서와 예시 전환 번호를 만듭니다.
# 모델 학습이나 정확도 검증은 하지 않으며, 배열 크기와 CPU/GPU 배치를 출력합니다.

import torch


# 예시 텐서를 만들고 배열 크기, 계산 장치, 영상 텐서의 메모리 크기를 출력합니다.
def main():
    # CUDA 사용이 가능하면 GPU, 그렇지 않으면 CPU에서 계산하도록 선택합니다.
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # 배치 = 한 번에 처리할 영상 수. 여기서는 영상 1개에서 16프레임을 다룬다고 가정합니다.
    batch_size = 1
    num_frames = 16
    # 색상 3채널, 높이/너비 224픽셀이라는 입력 크기를 가정합니다.
    channels = 3
    height = 224
    width = 224

    # 모양은 [배치, 시간(프레임), 채널, 높이, 너비]입니다.
    # randn은 표준정규분포의 무작위 수를 만들고 .to(device)는 선택한 장치로 텐서를 옮깁니다.
    video_tensor = torch.randn(
        batch_size,
        num_frames,
        channels,
        height,
        width
    ).to(device)

    # 영상의 각 프레임에 번호 하나를 대응시켜 [배치, 시간] 크기로 만듭니다. 초기값 0은 일반 장면입니다.
    scene_signal = torch.zeros(
        batch_size,
        num_frames
    ).to(device)

    # 인덱스는 0부터 시작하므로 첫 영상의 여섯 번째 프레임을 급격한 전환으로 표시합니다.
    scene_signal[0, 5] = 1
    # 슬라이스 끝 13은 제외됩니다. 인덱스 10, 11, 12의 세 프레임을 점진적 전환으로 표시합니다.
    scene_signal[0, 10:13] = 2

    print("-" * 60)
    print("Temporal Input Test")
    print("-" * 60)

    print("Device:", device)

    print(
        "Video tensor shape:",
        video_tensor.shape
    )

    print(
        "Scene signal shape:",
        scene_signal.shape
    )

    print(
        "Video tensor device:",
        video_tensor.device
    )

    print(
        "Scene signal:",
        scene_signal
    )

    # 원소 1개의 바이트 수 × 전체 원소 수로 영상 텐서만의 크기를 계산합니다.
    # 1024²로 나눈 값은 엄밀히 MiB이며, GPU 전체 사용량이나 모델 메모리를 뜻하지 않습니다.
    memory_mb = (
        video_tensor.element_size()
        * video_tensor.nelement()
        / 1024 ** 2
    )

    print(
        "Video tensor memory:",
        round(memory_mb, 2),
        "MB"
    )

    print("-" * 60)


# 다른 파일에서 import할 때는 실행하지 않고, 이 파일을 직접 실행했을 때만 아래 작업을 시작합니다.
if __name__ == "__main__":
    main()
