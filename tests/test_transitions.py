# ============================================================================
# 전환 합성 코드 테스트
# ----------------------------------------------------------------------------
# 테스트란
#   코드가 "이렇게 동작해야 한다"를 작은 예제로 확인하는 코드입니다.
#   코드를 고친 뒤 테스트를 돌리면, 실험 결과를 망가뜨리는 실수를 바로 알 수 있습니다.
#   예) 디졸브 섞는 비율 공식을 실수로 바꾸면 test_dissolve_is_linear_alpha_blend 가 실패합니다.
#
# 실행 (프로젝트 폴더에서)
#   .venv/Scripts/python.exe -m pytest tests
#
# pytest 기본 규칙
#   - tests 폴더에서 이름이 test_ 로 시작하는 파일과 함수를 찾아 자동으로 실행합니다.
#   - 함수 안의 assert 조건이 거짓이면 그 테스트는 실패로 표시됩니다.
#   - @pytest.mark.parametrize : 같은 테스트를 입력만 바꿔 여러 번 실행합니다.
#   - tmp_path : pytest 가 테스트마다 만들어 주는 임시 폴더입니다. 테스트가 끝나면 신경 쓰지 않아도 됩니다.
#   - @pytest.mark.skipif : 조건이 참이면(예: ffmpeg 가 없으면) 그 테스트를 건너뜁니다.
# ============================================================================

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest


# 테스트 파일은 tests 폴더에 있으므로 한 단계 위(프로젝트 폴더)를 import 경로에 넣어야 합니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.synthesize_transitions import (  # noqa: E402
    assign_partners,
    compose_transition,
    transition_window,
)
from utils.video_io import read_frames, write_frames  # noqa: E402


# 모든 픽셀이 같은 값인 작은 영상 배열을 만듭니다. 값을 미리 알고 있으니 결과를 계산으로 확인하기 쉽습니다.
#   solid(200) → 모양 [40, 4, 6, 3], 모든 값이 200
def solid(value, frames=40, height=4, width=6):
    return np.full((frames, height, width, 3), value, dtype=np.uint8)


# 전환 구간 밖의 프레임은 원래 영상에서 그대로 복사돼야 합니다.
# 이게 깨지면 "전환 구간 밖에서는 대조군과 원본이 같다"는 짝지은 비교의 전제가 무너집니다.
@pytest.mark.parametrize("kind, length", [("hard_cut", 0), ("fade", 8), ("dissolve", 16)])
def test_frames_outside_window_come_from_sources(kind, length):
    output = compose_transition(solid(200), solid(50), kind, length, center=20)
    start, end = transition_window(kind, length, center=20)

    assert (output[:start] == 200).all()
    assert (output[end:] == 50).all()


# 하드컷은 정확히 center 번 프레임부터 B 가 나와야 합니다(한 프레임 어긋나면 비교 위치가 틀어짐).
def test_hard_cut_switches_exactly_at_center():
    output = compose_transition(solid(200), solid(50), "hard_cut", 0, center=20)

    assert (output[19] == 200).all()
    assert (output[20] == 50).all()


# 디졸브 프레임이 (1 - p) x A + p x B 공식과 정확히 같은지 한 장씩 계산해서 비교합니다.
def test_dissolve_is_linear_alpha_blend():
    length = 8
    output = compose_transition(solid(200), solid(50), "dissolve", length, center=20)
    start, _ = transition_window("dissolve", length, center=20)

    for k in range(length):
        p = (k + 1) / (length + 1)
        assert (output[start + k] == np.rint((1 - p) * 200 + p * 50)).all()


# 같은 영상끼리 페이드하면 밝기가 가운데로 갈수록 검정에 가까워지고, 앞뒤가 대칭이어야 합니다.
def test_fade_dips_toward_black_symmetrically():
    length = 8
    output = compose_transition(solid(180), solid(180), "fade", length, center=20)
    start, end = transition_window("fade", length, center=20)
    # [start:end, 0, 0, 0] : 전환 구간의 각 프레임에서 왼쪽 위 픽셀의 R 값만 꺼냅니다.
    values = output[start:end, 0, 0, 0].astype(int).tolist()

    assert values == values[::-1]
    assert min(values) <= 180 * 2 / (length + 1) + 1
    assert max(values) < 180


# 전환 구간이 영상 밖으로 나가면 조용히 잘못 만들지 말고 오류를 내야 합니다.
# pytest.raises(ValueError): 이 블록 안에서 ValueError 가 "나야" 통과합니다.
def test_window_outside_video_raises():
    with pytest.raises(ValueError):
        compose_transition(solid(200, frames=10), solid(50, frames=10), "dissolve", 16, center=5)


# 짝짓기 테스트에 쓸 영상 정보를 간단히 만드는 함수입니다.
def info(height, width, fps):
    return {"height": height, "width": width, "fps": fps}


# 해상도와 fps 가 같은 영상끼리 먼저 짝지어야 합니다. 짝이 없는 영상끼리는 서로 짝이 됩니다.
def test_partners_prefer_same_resolution_and_fps():
    infos = {
        "a": info(720, 1280, 30.0),
        "b": info(720, 1280, 30.0),
        "c": info(480, 854, 25.0),
        "d": info(1080, 1920, 30.0),
    }
    partners = assign_partners(infos)

    assert partners["a"] == "b" and partners["b"] == "a"
    assert partners["c"] == "d" and partners["d"] == "c"
    assert set(partners) == set(infos)


# fps 까지 같은 짝이 없으면 해상도만 같은 영상끼리 짝지어야 합니다.
def test_partners_fall_back_to_same_resolution_when_fps_differs():
    infos = {
        "a": info(720, 1280, 30.0),
        "b": info(720, 1280, 25.0),
        "c": info(720, 1280, 30.0),
        "d": info(720, 1280, 29.97),
    }
    partners = assign_partners(infos)

    # a, c 는 해상도와 fps 가 같아 서로 짝이 되고, 남은 b, d 는 해상도만 같아서 짝이 됩니다.
    assert partners["a"] == "c" and partners["c"] == "a"
    assert partners["b"] == "d" and partners["d"] == "b"


# 영상을 저장했다가 다시 읽었을 때 프레임 개수와 순서가 그대로여야 합니다.
# 프레임이 하나라도 빠지거나 복제되면 전환을 넣은 위치와 채점 위치가 어긋나 실험이 무의미해집니다.
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg 필요")
def test_video_roundtrip_keeps_frame_count_and_order(tmp_path):
    # 프레임마다 밝기를 다르게 해서(10, 19, 28, ...) 다시 읽었을 때 개수와 순서가 그대로인지 확인합니다.
    count = 24
    frames = np.stack([np.full((32, 32, 3), 10 + 9 * k, dtype=np.uint8) for k in range(count)])
    path = tmp_path / "roundtrip.mp4"

    write_frames(path, frames, fps=30.0, crf=18)
    decoded, fps = read_frames(path)

    assert len(decoded) == count
    assert abs(fps - 30.0) < 1e-6
    # 프레임마다 평균 밝기를 구해, 계속 커지는지(순서 유지)와 원래 값과 거의 같은지(압축 오차 4 미만)를 봅니다.
    means = decoded.reshape(count, -1).mean(axis=1)
    assert np.all(np.diff(means) > 0)
    assert np.abs(means - frames.reshape(count, -1).mean(axis=1)).max() < 4
