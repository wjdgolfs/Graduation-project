# ============================================================================
# 파일럿용 전환 합성 영상 만들기
# ----------------------------------------------------------------------------
# 무엇을 하나
#   real 영상 A 와 다른 real 영상 B 를 이어 붙이면서, 그 사이에 장면 전환(하드컷/페이드/디졸브)을 넣습니다.
#   두 영상 모두 real 이므로, 탐지기가 이 영상을 fake 로 판단하면 그건 전부 "오탐(false positive)"입니다.
#
# 입력 / 출력
#   입력: FF++ original c23 영상 중 공식 분할(기본 test)에 속한 영상
#   출력: processed/pilot/clips/*.mp4    ← 대조군 영상과 전환 합성 영상
#         processed/pilot/manifest.csv   ← 어떤 영상을 어떻게 만들었는지 적은 목록
#
# 실행 예 (프로젝트 폴더에서)
#   python preprocessing/synthesize_transitions.py                  설정값(pilot.num_videos)만큼 만들기
#   python preprocessing/synthesize_transitions.py --num-videos 30  영상 30개로 만들기
#   python preprocessing/synthesize_transitions.py --prune          새 목록에 없는 옛 영상 지우기
#
# 세 가지 전환 (p 는 전환 진행도: 0 에 가까우면 A, 1 에 가까우면 B)
#   hard_cut : 한 프레임 만에 A 에서 B 로 바뀝니다.
#   dissolve : 출력 = (1 - p) x A + p x B       ← 두 이미지를 섞는 "알파 블렌딩"
#   fade     : 앞 절반은 A 가 검게 어두워지고 (A x (1 - 2p)),
#              뒤 절반은 검은 화면에서 B 가 밝아집니다 (B x (2p - 1)).
#   디졸브의 알파 블렌딩은 face-swap 이 가짜 얼굴의 경계를 섞을 때 쓰는 연산과 같은 형태입니다.
#   그래서 시간 기반 탐지기가 디졸브를 조작 흔적으로 착각할 것이라는 게 이 연구의 가설입니다.
#
# 왜 test 분할만 쓰는가
#   FTCN 은 FF++ train 분할로 학습됐습니다. 학습에 쓴 real 영상은 점수가 낮게 나오기 쉬워서
#   오탐이 실제보다 적게 잡힙니다. 탐지기가 본 적 없는 영상으로 재야 공정합니다.
#
# 프레임 번호 규칙 (짝지은 비교의 핵심)
#   영상마다 앞에서부터 segment_frames(256) 장을 잘라 씁니다.
#   전환 구간 [window_start, window_end) 앞은 A 의 프레임, 뒤는 B 의 "같은 번호" 프레임입니다.
#   예) 64프레임 디졸브, 중심 128
#
#       프레임 번호 :  0 ........... 95 [ 96 ~ 159 전환 구간 ] 160 ........... 255
#       합성본      :  A0 ........... A95   A 와 B 가 섞인 프레임   B160 ......... B255
#       대조군 A    :  A0 ...................................................... A255
#       대조군 B    :  B0 ...................................................... B255
#
#   그래서 전환 구간 밖의 프레임은 대조군의 같은 번호 프레임과 원본이 같고,
#   탐지 점수의 차이는 전환 구간과 그 주변(탐지기가 한 번에 보는 프레임 수만큼)에서만 생겨야 합니다.
# ============================================================================

import argparse  # 명령줄 옵션(--num-videos 등)을 읽습니다.
import csv  # manifest.csv 를 씁니다.
import json  # FF++ 분할 파일(json)을 읽습니다.
import sys  # import 경로(sys.path)를 바꿀 때 씁니다.
from pathlib import Path

import cv2  # partner 영상의 크기를 맞출 때(resize) 씁니다.
import numpy as np
import yaml  # configs/config.yaml 을 읽습니다.


# __file__는 현재 파일 경로입니다. 상위 두 단계인 프로젝트 폴더를 기준으로 설정 파일을 찾습니다.
#   parents[0] = preprocessing 폴더, parents[1] = 프로젝트 폴더
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# 다른 폴더의 공용 코드(utils)를 불러오려면 프로젝트 폴더가 import 경로에 있어야 합니다.
# "python preprocessing/synthesize_transitions.py" 로 실행하면 파이썬은 preprocessing 폴더만 경로에 넣기 때문입니다.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 위에서 경로를 넣은 뒤에 import 해야 해서 파일 맨 위가 아니라 여기에 둡니다.
# "# noqa: E402" 는 "import 가 파일 위쪽에 없다"는 코드 검사기 경고를 끄는 표시입니다.
from utils.video_io import probe_video, read_frames, write_frames  # noqa: E402


# FF++ 폴더 구조. config.yaml 의 raw 경로 아래 기준입니다.
FFPP_DIR = Path("FaceForensics++")
FFPP_ORIGINAL_DIR = FFPP_DIR / "original_sequences" / "youtube" / "c23" / "videos"
# FF++ 공식 분할 파일. https://github.com/ondyari/FaceForensics/tree/master/dataset/splits
FFPP_SPLIT_DIR = FFPP_DIR / "splits"

TRANSITION_KINDS = ("hard_cut", "fade", "dissolve")

# manifest.csv 의 열 이름입니다. 한 줄이 영상 하나입니다.
#   clip_id         : 영상 이름. 합성본은 "A번호__전환종류_길이__B번호", 대조군은 "A번호__control"
#   role            : control(대조군, 전환 없음) 또는 treatment(전환 합성본)
#   base / partner  : A(앞 영상) / B(뒤 영상)의 FF++ 번호
#   kind / length   : 전환 종류와 길이(프레임 수)
#   center          : 전환 중심 프레임 번호
#   window_start / window_end : 전환 구간 [시작, 끝). 끝 번호의 프레임은 구간에 포함되지 않습니다.
#   segment_frames  : 영상마다 잘라 쓴 프레임 수
#   fps, width, height : A 영상의 fps 와 크기 (B 는 이 크기로 맞춥니다)
#   partner_resized : B 의 크기를 바꿨는지
#   fps_mismatch    : A 와 B 의 fps 가 다른지. 프레임 번호로 합성하므로 B 의 움직임 속도가 조금 달라 보일 수 있습니다.
#   path            : 저장된 mp4 경로
MANIFEST_COLUMNS = [
    "clip_id", "role", "base", "partner", "kind", "length",
    "center", "window_start", "window_end", "segment_frames",
    "fps", "width", "height", "partner_resized", "fps_mismatch", "path",
]


# 여러 스크립트가 같은 경로와 기준값을 쓰도록 YAML 설정을 딕셔너리로 읽습니다.
def load_config():
    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return yaml.safe_load(file)


# FF++ 공식 분할에 속한 원본 영상 번호를 이름순으로 돌려줍니다.
# 분할 파일은 [["953", "974"], ...] 처럼 원본 영상 두 개씩 짝지은 목록입니다.
#   FF++ 는 이 두 영상끼리 얼굴을 바꿔 조작 영상 "953_974", "974_953" 을 만들었습니다.
#   test 분할은 70쌍이라 원본 영상은 140개입니다.
def load_split_ids(raw_dir, split):
    path = Path(raw_dir) / FFPP_SPLIT_DIR / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"FF++ 분할 파일이 없습니다: {path}\n"
            "https://github.com/ondyari/FaceForensics/tree/master/dataset/splits 의 json 을 이 폴더에 두세요."
        )
    with open(path, "r", encoding="utf-8") as file:
        pairs = json.load(file)
    # 이중 for 문을 한 줄로 쓴 "집합 내포(set comprehension)"입니다.
    # 쌍 [a, b] 를 펼쳐 a 와 b 를 모두 모으고, 집합({ })이라 중복은 저절로 사라집니다.
    return sorted({name for pair in pairs for name in pair})


# 전환 구간의 시작 번호와 끝 번호를 돌려줍니다. 끝 번호의 프레임은 구간에 포함되지 않습니다.
# hard_cut 은 길이가 0 이라 시작과 끝이 같고, center 번 프레임부터 B 가 나옵니다.
#   예) transition_window("dissolve", 32, 128) → (112, 144) : 112~143번 프레임이 섞인 프레임
def transition_window(kind, length, center):
    if kind not in TRANSITION_KINDS:
        raise ValueError(f"알 수 없는 전환 종류입니다: {kind}")

    if kind == "hard_cut":
        length = 0

    # "//" 는 몫만 남기는 나눗셈입니다. 32 // 2 = 16 이라 중심 앞뒤로 16프레임씩 나뉩니다.
    start = center - length // 2
    return start, start + length


# base(A) 에서 partner(B) 로 넘어가는 프레임 배열을 만듭니다.
# 두 입력은 모양이 같은 [N, H, W, 3] uint8 배열이어야 하고, 반환값도 같은 모양입니다.
def compose_transition(base, partner, kind, length, center):
    if base.shape != partner.shape:
        raise ValueError(f"두 영상의 배열 모양이 같아야 합니다: {base.shape} / {partner.shape}")

    start, end = transition_window(kind, length, center)
    if start < 0 or end > len(base):
        raise ValueError(f"전환 구간 [{start}, {end}) 이 영상 길이 {len(base)} 를 벗어납니다.")

    # 결과를 담을 배열. empty_like 는 base 와 같은 모양·자료형으로 메모리만 잡고 값은 채우지 않습니다.
    output = np.empty_like(base)
    # 전환 구간 앞은 A 그대로, 뒤는 B 그대로 복사합니다(같은 프레임 번호끼리).
    output[:start] = base[:start]
    output[end:] = partner[end:]

    count = end - start
    # 전환 구간 안의 프레임을 하나씩 만듭니다. hard_cut 은 count 가 0 이라 이 반복을 건너뜁니다.
    for k in range(count):
        # 진행도 p 는 0 과 1 을 뺀 값입니다. 순수한 A, B 프레임은 구간 바로 앞뒤에 이미 있기 때문입니다.
        #   예) count = 8 이면 p = 1/9, 2/9, ..., 8/9
        p = (k + 1) / (count + 1)
        # 섞는 계산에서 소수가 나오므로 실수(float32)로 바꿔서 계산합니다.
        # uint8 끼리 바로 더하면 255 를 넘는 순간 0 부터 다시 세는 넘침도 생길 수 있습니다(예: 200 + 100 → 44).
        a = base[start + k].astype(np.float32)
        b = partner[start + k].astype(np.float32)

        if kind == "dissolve":
            # 두 장면의 알파 블렌딩. face-swap 경계를 만드는 블렌딩과 같은 형태의 연산입니다.
            mixed = (1.0 - p) * a + p * b
        elif p < 0.5:
            # fade 앞 절반: A 가 검은 화면 쪽으로 어두워집니다(p 가 0.5 에 가까울수록 검정에 가까움).
            mixed = a * (1.0 - 2.0 * p)
        else:
            # fade 뒤 절반: 검은 화면에서 B 가 밝아집니다.
            mixed = b * (2.0 * p - 1.0)

        # 반올림(rint) → 0~255 범위로 자르기(clip) → 다시 uint8 로 바꿔 저장합니다.
        output[start + k] = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)

    return output


# partner 프레임을 base 와 같은 해상도로 맞춥니다. 반환값은 (프레임 배열, 크기를 바꿨는지) 입니다.
def match_resolution(frames, height, width):
    # frames.shape[1:3] 은 (세로, 가로) 입니다. 이미 같으면 그대로 돌려줍니다.
    if frames.shape[1:3] == (height, width):
        return frames, False

    # 줄일 때는 INTER_AREA 가 계단 현상이 적고, 키울 때는 INTER_LINEAR 가 무난합니다.
    interpolation = cv2.INTER_AREA if frames.shape[1] > height else cv2.INTER_LINEAR
    # 주의: cv2.resize 는 크기를 (가로, 세로) 순서로 받습니다. numpy 의 (세로, 가로) 와 반대입니다.
    resized = np.stack([
        cv2.resize(frame, (width, height), interpolation=interpolation)
        for frame in frames
    ])
    return resized, True


# 영상마다 전환 상대(partner)를 정합니다. 조건이 비슷한 영상끼리 먼저 짝짓습니다.
#   1순위: 해상도와 fps 가 모두 같은 영상끼리
#   2순위: 해상도만 같은 영상끼리
#   남은 영상: 서로 짝짓고 크기를 맞춥니다
# 각 묶음 안에서는 이름순으로 한 칸씩 밀어 짝짓습니다(A→B, B→C, ..., 마지막→A).
#   예) 묶음 [015, 036, 128] → 015 의 상대 036, 036 의 상대 128, 128 의 상대 015
# infos 는 {영상 이름: {"height": ..., "width": ..., "fps": ...}} 모양의 딕셔너리입니다.
def assign_partners(infos):
    if len(infos) < 2:
        raise ValueError("전환 상대를 정하려면 영상이 2개 이상 필요합니다.")

    # lambda 는 이름 없는 짧은 함수입니다. 영상 정보를 받아 "같은 묶음인지 판단할 값"을 튜플로 돌려줍니다.
    group_keys = [
        lambda info: (info["height"], info["width"], round(info["fps"], 2)),
        lambda info: (info["height"], info["width"]),
    ]

    partners = {}
    remaining = sorted(infos)

    for key in group_keys:
        # 기준값이 같은 영상끼리 묶습니다. setdefault 는 키가 없으면 빈 리스트를 만들어 준 뒤 돌려줍니다.
        groups = {}
        for name in remaining:
            groups.setdefault(key(infos[name]), []).append(name)

        remaining = []
        for names in groups.values():
            # 묶음에 혼자 남은 영상은 다음 순위 기준으로 다시 묶어 봅니다.
            if len(names) == 1:
                remaining.extend(names)
                continue
            # "% len(names)" 는 나머지 연산이라, 마지막 영상의 다음은 다시 첫 영상이 됩니다.
            for index, name in enumerate(names):
                partners[name] = names[(index + 1) % len(names)]
        remaining.sort()

    # 끝까지 짝을 못 찾은 영상들 처리
    if len(remaining) == 1:
        # 혼자뿐이면 이름순으로 첫 번째 다른 영상과 짝짓습니다.
        others = sorted(name for name in infos if name != remaining[0])
        partners[remaining[0]] = others[0]
    else:
        for index, name in enumerate(remaining):
            partners[name] = remaining[(index + 1) % len(remaining)]

    return partners


# 분할에 속하고 이미 받아져 있는 영상 중, 이름순 앞에서부터 num_videos 개를 고릅니다.
# 반환값: (고른 영상 정보 딕셔너리, 아직 받아지지 않은 분할 영상 이름 목록)
def select_videos(source_dir, split_ids, num_videos, segment):
    # 폴더의 mp4 파일을 {이름: 경로} 로 모읍니다. path.stem 은 확장자를 뺀 이름입니다("035.mp4" → "035").
    local = {path.stem: path for path in source_dir.glob("*.mp4")}
    missing = [name for name in split_ids if name not in local]

    infos = {}
    for name in split_ids:
        if name not in local:
            continue
        info = probe_video(local[name])
        # segment(256)장보다 짧은 영상은 앞에서 256장을 잘라낼 수 없으므로 뺍니다.
        if info["frames"] and info["frames"] < segment:
            print(f"건너뜀: {name} (프레임 {info['frames']}장 < {segment}장)")
            continue
        # {**info, "path": ...} 는 info 딕셔너리를 복사하면서 "path" 키를 하나 더 넣는 문법입니다.
        infos[name] = {**info, "path": local[name]}
        if len(infos) == num_videos:
            break

    return infos, missing


def main():
    # 명령줄 옵션을 정의합니다. 실행할 때 "--overwrite" 처럼 붙이면 args.overwrite 가 True 가 됩니다.
    parser = argparse.ArgumentParser(
        description="파일럿용 대조군 영상과 전환 합성 영상을 만듭니다."
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        default=None,
        help="config.yaml 의 pilot.num_videos 대신 쓸 영상 수"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="이미 만든 영상도 다시 만듭니다."
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="새 manifest 에 없는 옛 합성 영상을 clips 폴더에서 지웁니다."
    )
    args = parser.parse_args()

    config = load_config()
    pilot = config["pilot"]

    # 입력 폴더(FF++ 원본)와 출력 폴더(processed/pilot) 경로
    source_dir = Path(config["paths"]["raw"]) / FFPP_ORIGINAL_DIR
    output_dir = Path(config["paths"]["processed"]) / "pilot"
    clip_dir = output_dir / "clips"

    segment = pilot["segment_frames"]
    center = pilot["transition_center"]
    # "a or b" 는 a 가 None(또는 0)이면 b 를 씁니다. 옵션을 안 주면 설정 파일 값을 씁니다.
    num_videos = args.num_videos or pilot["num_videos"]

    # 1) 쓸 영상 고르기
    split_ids = load_split_ids(config["paths"]["raw"], pilot["split"])
    infos, missing = select_videos(source_dir, split_ids, num_videos, segment)

    print(f"{pilot['split']} 분할 영상 {len(split_ids)}개 중 받아진 것 {len(split_ids) - len(missing)}개")
    print(f"사용할 영상 {len(infos)}개: {', '.join(sorted(infos))}")
    if len(infos) < num_videos:
        print(f"주의: 요청한 {num_videos}개보다 적습니다. 다운로드가 끝나면 다시 실행하세요.")
    if infos:
        # 고른 영상보다 이름이 앞서는데 아직 안 받아진 영상이 있으면, 나중에 다시 실행할 때 선택이 바뀝니다.
        earlier_missing = [name for name in missing if name < max(infos)]
        if earlier_missing:
            print(
                f"주의: 이름순으로 앞선 {len(earlier_missing)}개가 아직 받아지지 않아, "
                f"다운로드 후 다시 실행하면 선택이 바뀔 수 있습니다: {earlier_missing[:8]}"
            )

    # 2) 영상마다 전환 상대 정하기
    partners = assign_partners(infos)

    # 3) 영상마다 대조군 1개 + 전환 합성본(설정의 transitions 개수만큼) 만들기
    rows = []
    for name in sorted(infos):
        partner_name = partners[name]
        base, fps = read_frames(infos[name]["path"], start=0, count=segment)
        partner, partner_fps = read_frames(infos[partner_name]["path"], start=0, count=segment)

        if len(base) < segment or len(partner) < segment:
            print(f"건너뜀: {name} (디코딩된 프레임이 {segment}장보다 적음)")
            continue

        # fps 는 소수라 == 로 비교하지 않고, 차이가 아주 작으면 같다고 봅니다.
        fps_mismatch = abs(partner_fps - fps) > 0.01
        height, width = base.shape[1:3]
        partner, resized = match_resolution(partner, height, width)

        # 대조군과 합성본 행에 공통으로 들어갈 값
        common = {
            "base": name,
            "center": center,
            "segment_frames": segment,
            "fps": fps,
            "width": width,
            "height": height,
        }

        # 3-1) 대조군: A 를 그대로 다시 압축합니다. 합성본과 같은 압축 조건을 거치게 하려는 것입니다.
        control_id = f"{name}__control"
        control_path = clip_dir / f"{control_id}.mp4"
        # 이미 만든 파일은 다시 만들지 않습니다(--overwrite 를 주면 다시 만듦).
        if args.overwrite or not control_path.exists():
            write_frames(control_path, base, fps, crf=pilot["crf"], preset=pilot["preset"])
        rows.append({
            **common,
            "clip_id": control_id,
            "role": "control",
            "partner": "",
            "kind": "none",
            "length": 0,
            "window_start": center,
            "window_end": center,
            "partner_resized": False,
            "fps_mismatch": False,
            # as_posix(): 윈도우의 "\" 대신 "/" 로 경로를 적어 다른 운영체제에서도 읽기 쉽게 합니다.
            "path": control_path.as_posix(),
        })

        # 3-2) 전환 합성본: 설정의 전환 종류 × 길이마다 하나씩
        total = 0
        made = 0
        for kind, lengths in pilot["transitions"].items():
            for length in lengths:
                window_start, window_end = transition_window(kind, length, center)
                # {length:02d} 는 두 자리로 맞춰 적는 형식입니다. 8 → "08"
                clip_id = f"{name}__{kind}_{length:02d}__{partner_name}"
                clip_path = clip_dir / f"{clip_id}.mp4"

                if args.overwrite or not clip_path.exists():
                    frames = compose_transition(base, partner, kind, length, center)
                    write_frames(clip_path, frames, fps, crf=pilot["crf"], preset=pilot["preset"])
                    made += 1
                total += 1

                rows.append({
                    **common,
                    "clip_id": clip_id,
                    "role": "treatment",
                    "partner": partner_name,
                    "kind": kind,
                    "length": window_end - window_start,
                    "window_start": window_start,
                    "window_end": window_end,
                    "partner_resized": resized,
                    "fps_mismatch": fps_mismatch,
                    "path": clip_path.as_posix(),
                })

        notes = [note for note, flag in (("크기 조정함", resized), ("fps 다름", fps_mismatch)) if flag]
        suffix = f" ({', '.join(notes)})" if notes else ""
        print(f"{name}: 전환 {total}개 (새로 {made}개), partner={partner_name}{suffix}")

    # 4) manifest.csv 저장. DictWriter 는 딕셔너리 목록을 열 이름에 맞춰 CSV 로 씁니다.
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"manifest 저장: {manifest_path} ({len(rows)}줄)")

    # 5) (선택) 영상 선택이 바뀌어 새 목록에 없는 옛 합성 영상을 지웁니다. 지운 파일은 되돌릴 수 없습니다.
    if args.prune:
        keep = {row["clip_id"] for row in rows}
        stale = [path for path in clip_dir.glob("*.mp4") if path.stem not in keep]
        for path in stale:
            path.unlink()
        print(f"manifest 에 없는 옛 합성 영상 {len(stale)}개를 지웠습니다.")


# 이 파일을 직접 실행할 때만 main() 을 부릅니다. 다른 파일(테스트 등)에서 import 할 때는 실행되지 않습니다.
if __name__ == "__main__":
    main()
