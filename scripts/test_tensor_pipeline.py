import torch


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    batch_size = 1
    num_frames = 16
    channels = 3
    height = 224
    width = 224

    video_tensor = torch.randn(
        batch_size,
        num_frames,
        channels,
        height,
        width
    ).to(device)

    scene_signal = torch.zeros(
        batch_size,
        num_frames
    ).to(device)

    scene_signal[0, 5] = 1
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


if __name__ == "__main__":
    main()