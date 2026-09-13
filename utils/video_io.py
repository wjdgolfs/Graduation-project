# ============================================================================
# 영상 입출력 공용 함수
# ----------------------------------------------------------------------------
# 이 파일이 하는 일
#   1) probe_video  : 영상 파일의 기본 정보(프레임 수, fps, 가로·세로 크기)를 읽습니다.
#   2) read_frames  : 영상을 프레임(이미지) 배열로 읽습니다.
#   3) write_frames : 프레임 배열을 H.264 mp4 영상으로 저장합니다.
#
# 알아두면 좋은 개념
#   - 프레임: 영상을 이루는 이미지 한 장. 30fps 영상은 1초에 30장입니다.
#   - 프레임 배열: numpy 배열 모양 [N, H, W, 3]
#       N = 프레임 수, H = 세로(높이) 픽셀 수, W = 가로(너비) 픽셀 수, 3 = 색 채널(R, G, B)
#       값 하나하나는 uint8(0~255 정수)입니다.
#       예) 1280x720 영상 256장 = 256 x 720 x 1280 x 3 바이트 ≈ 700MB 를 메모리에 올립니다.
#   - 컨테이너와 코덱: mp4 는 "상자(컨테이너)"이고, 안에 든 영상은 H.264 같은 "압축 방식(코덱)"으로 저장됩니다.
#   - CRF: H.264 압축 강도. 숫자가 클수록 더 많이 압축합니다(화질은 낮아짐).
#       FF++ 의 "c23" 은 CRF 23 으로 압축한 버전이라는 뜻입니다.
#
# 왜 읽기와 쓰기에 서로 다른 도구를 쓰는가
#   - 읽기(PyAV): 디코더가 내놓는 순서대로 프레임을 하나씩 받아서 "몇 번째 프레임인지"가 정확합니다.
#     이 연구는 "128번 프레임을 중심으로 32프레임 디졸브를 넣었다"처럼 프레임 번호가 실험의 전제라서 중요합니다.
#   - 쓰기(ffmpeg 프로그램): libx264 의 CRF 값을 직접 지정할 수 있어서, FF++ c23 과 같은 조건으로 다시 압축할 수 있습니다.
# ============================================================================

import shutil  # 컴퓨터에 설치된 프로그램(ffmpeg)의 위치를 찾는 데 씁니다.
import subprocess  # 파이썬에서 다른 프로그램(ffmpeg)을 실행하고 데이터를 주고받습니다.
from fractions import Fraction  # 29.97 같은 fps 를 30000/1001 같은 분수로 정확하게 다룹니다.
from pathlib import Path  # 파일 경로를 문자열 대신 객체로 다루면 폴더 만들기·경로 합치기가 편합니다.

import av  # PyAV: FFmpeg 라이브러리를 파이썬에서 직접 쓰게 해 주는 패키지입니다.
import numpy as np


# 영상 파일의 헤더(머리말) 정보를 읽어 딕셔너리로 돌려줍니다. 프레임을 풀지 않아서 빠릅니다.
#   반환 예: {"frames": 423, "fps": 30.0, "width": 1280, "height": 720}
# 주의: frames 는 파일에 "적혀 있는" 값이라, 실제로 디코딩되는 프레임 수와 드물게 다를 수 있습니다.
def probe_video(path):
    # with 문: 블록이 끝나면(중간에 오류가 나도) 파일을 자동으로 닫아 줍니다.
    with av.open(str(path)) as container:
        # 컨테이너 안에는 영상·음성 스트림이 여러 개 있을 수 있어서 첫 번째 영상 스트림을 고릅니다.
        stream = container.streams.video[0]
        return {
            "frames": stream.frames,
            # average_rate 는 분수(Fraction)라 float 로 바꿉니다. 예) 30000/1001 → 29.97
            "fps": float(stream.average_rate),
            "width": stream.codec_context.width,
            "height": stream.codec_context.height,
        }


# start 번 프레임부터 count 장을 읽어 RGB 배열 [N, H, W, 3] (uint8) 로 돌려줍니다.
# count 가 None 이면 영상 끝까지 읽습니다. 반환값은 (프레임 배열, fps) 두 개입니다.
#   예) frames, fps = read_frames("035.mp4", start=0, count=256)
#       → frames.shape == (256, 720, 1280, 3), fps == 30.0
def read_frames(path, start=0, count=None):
    frames = []

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        # 디코딩(압축 풀기)을 여러 CPU 스레드로 나눠 빠르게 합니다. 결과로 나오는 프레임 순서는 바뀌지 않습니다.
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate)

        # container.decode(stream) 은 프레임을 화면에 보여줄 순서대로 하나씩 내놓습니다.
        # enumerate 는 (번호, 값) 쌍을 만들어 주므로, index 가 곧 0부터 센 프레임 번호입니다.
        for index, frame in enumerate(container.decode(stream)):
            # 시작 번호 전의 프레임은 버립니다.
            # H.264 는 앞 프레임을 참고해서 압축하기 때문에, 앞에서부터 차례로 풀면서 건너뜁니다.
            if index < start:
                continue
            # 필요한 장수를 다 읽었으면 나머지는 풀지 않고 멈춥니다.
            if count is not None and index >= start + count:
                break
            # 디코더는 YUV 라는 색 형식으로 풀어 주므로, 사람이 보는 RGB 로 바꿔 numpy 배열로 받습니다.
            frames.append(frame.to_ndarray(format="rgb24"))

    if not frames:
        raise ValueError(f"프레임을 읽지 못했습니다: {path} (start={start}, count={count})")

    # np.stack: [H, W, 3] 배열 N 개를 쌓아서 [N, H, W, 3] 배열 하나로 만듭니다.
    return np.stack(frames), fps


# 프레임 배열 [N, H, W, 3] (uint8, RGB) 를 H.264 mp4 로 저장합니다.
#   crf    : 압축 강도. 23 은 FF++ c23 과 같은 값입니다.
#   preset : 인코딩 속도와 압축 효율의 균형. medium 이 ffmpeg 기본값입니다.
# 파일럿에서는 대조군과 전환 합성본을 모두 이 함수로 같은 조건에서 다시 압축합니다.
# 그래야 두 영상의 점수 차이가 "압축을 한 번 더 했는지"가 아니라 "전환이 들어갔는지"에서만 나옵니다.
def write_frames(path, frames, fps, crf=23, preset="medium"):
    # PATH 환경 변수에 등록된 폴더들에서 ffmpeg 실행 파일을 찾습니다. 없으면 None 입니다.
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg 를 찾을 수 없습니다. PATH 를 확인하세요.")

    # 리스트가 들어와도 numpy 배열로 바꿔서 모양을 검사합니다.
    frames = np.asarray(frames)
    # ndim = 차원 수(4 여야 함), shape[3] = 채널 수(3 이어야 함), dtype = 값의 자료형(uint8 이어야 함)
    if frames.ndim != 4 or frames.shape[3] != 3 or frames.dtype != np.uint8:
        raise ValueError(
            f"[N, H, W, 3] uint8 배열이어야 합니다: shape={frames.shape}, dtype={frames.dtype}"
        )

    # yuv420p 형식은 색 정보를 가로·세로 절반 해상도로 저장해서 크기가 짝수여야 합니다.
    # "크기 % 2" 는 홀수면 1, 짝수면 0 이라서, 홀수일 때만 1픽셀 줄어듭니다.
    height = frames.shape[1] - frames.shape[1] % 2
    width = frames.shape[2] - frames.shape[2] % 2
    # [:, :height, :width] = 모든 프레임에서 위쪽 height 줄, 왼쪽 width 칸만 남기는 슬라이싱입니다.
    frames = frames[:, :height, :width]

    path = Path(path)
    # 저장할 폴더가 없으면 만듭니다. parents=True 는 중간 폴더까지, exist_ok=True 는 이미 있어도 오류를 내지 않습니다.
    path.parent.mkdir(parents=True, exist_ok=True)
    # 30.0 → 30/1, 29.97002997 → 30000/1001 처럼 fps 를 분수로 바꿉니다(분모는 1001 이하로 제한).
    rate = Fraction(fps).limit_denominator(1001)

    # ffmpeg 에 넘길 명령어입니다. 터미널에서 직접 입력하는 명령과 같습니다.
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",     # 안내 문구 숨김, 오류만 출력, 같은 이름 파일 덮어쓰기
        "-f", "rawvideo", "-pix_fmt", "rgb24",                   # 입력은 압축 안 된 RGB 바이트
        "-s", f"{width}x{height}",                                # 입력 한 장의 크기 (가로x세로)
        "-r", f"{rate.numerator}/{rate.denominator}",             # 입력 fps
        "-i", "-",                                                # 입력을 파일이 아니라 표준 입력(파이프)으로 받음
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),   # H.264 로 압축
        "-pix_fmt", "yuv420p",                                    # 대부분의 플레이어가 재생할 수 있는 색 형식
        str(path),                                                # 출력 파일
    ]

    # Popen 은 ffmpeg 를 실행해 두고 바로 돌아옵니다. stdin=PIPE 로 우리가 데이터를 흘려 넣을 통로를 엽니다.
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            # 프레임 한 장을 바이트로 바꿔 ffmpeg 에 보냅니다.
            # 슬라이싱한 배열은 메모리에서 연속으로 붙어 있지 않을 수 있어서 ascontiguousarray 로 정리합니다.
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
        # 입력이 끝났다고 알려야 ffmpeg 가 압축을 마무리하고 파일을 닫습니다.
        process.stdin.close()
    except BrokenPipeError:
        # ffmpeg 가 먼저 종료된 경우입니다. 원인은 아래에서 stderr 로 보여줍니다.
        pass

    # ffmpeg 가 남긴 오류 메시지를 읽고, 종료 코드가 0(성공)이 아니면 오류를 냅니다.
    stderr = process.stderr.read().decode("utf-8", "replace")
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg 인코딩 실패: {path}\n{stderr.strip()}")


# 지정한 프레임 번호들만 골라 RGB 배열 [N, H, W, 3] (uint8) 로 읽습니다.
# 영상 전체를 메모리에 올리지 않고, 앞에서부터 풀면서 필요한 번호의 프레임만 남깁니다.
#   예) read_selected_frames("a.mp4", [0, 100, 200]) → 0, 100, 200번 프레임 3장
# 반환값: (프레임 배열, fps, 실제로 읽은 프레임 번호 배열)
#   번호는 작은 순서로 정렬되고 중복은 하나로 합칩니다. 영상이 짧아서 없는 번호는 빠집니다.
def read_selected_frames(path, indices):
    wanted = sorted({int(index) for index in indices})
    if not wanted:
        raise ValueError("읽을 프레임 번호가 없습니다.")
    # 집합(set)은 "이 번호가 들어 있나?" 확인이 리스트보다 훨씬 빠릅니다.
    wanted_set = set(wanted)
    frames = []
    found = []

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate)

        for index, frame in enumerate(container.decode(stream)):
            # 가장 큰 번호를 지나면 더 풀 필요가 없습니다.
            if index > wanted[-1]:
                break
            if index in wanted_set:
                frames.append(frame.to_ndarray(format="rgb24"))
                found.append(index)

    if not frames:
        raise ValueError(f"프레임을 읽지 못했습니다: {path}")

    return np.stack(frames), fps, np.array(found, dtype=np.int64)
