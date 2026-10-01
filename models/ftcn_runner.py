# ============================================================================
# FTCN 실행기: 영상 → 프레임별 딥페이크 점수
# ----------------------------------------------------------------------------
# FTCN 이란
#   "Exploring Temporal Coherence for More General Video Face Forgery Detection" (ICCV 2021) 의 모델입니다.
#   3D ResNet-50 의 합성곱에서 공간 방향 커널 크기를 1 로 줄여서, 한 위치가 "시간에 따라 어떻게 변하는지"만
#   보게 만들고(Fully Temporal Convolution), 그 위에 시간 축 Transformer 를 얹었습니다.
#   얼굴 모양이 아니라 시간적인 부자연스러움으로 판별하므로, 이 연구에서 "시간 기반 탐지기"의 대표로 씁니다.
#
# 이 파일이 영상 하나를 처리하는 순서 (score_video 함수)
#   1) 프레임 읽기       : OpenCV 로 프레임을 읽고 BGR → RGB 로 바꿉니다.
#   2) 얼굴 검출         : RetinaFace 로 프레임마다 얼굴 상자와 5점(두 눈, 코, 입 양끝)을 찾습니다.
#   3) 68점 랜드마크     : 얼굴마다 턱선·눈썹·코·눈·입 68개 점을 찾습니다.
#   4) 얼굴 추적         : 이웃 프레임의 얼굴 상자가 충분히 겹치면(IoU ≥ 0.5) 같은 얼굴로 이어 붙입니다.
#                          끊기면 끊긴 곳을 경계로 구간을 나누고, 구간마다 따로 채점합니다.
#   5) 정렬된 얼굴 크롭   : 5점을 표준 위치에 맞춰 얼굴을 224x224 로 잘라냅니다.
#   6) 32프레임 클립 채점 : 연속 32프레임을 한 묶음(클립)으로 모델에 넣어 fake 확률(0~1)을 얻습니다.
#                          클립은 1프레임씩 밀면서 만듭니다(256프레임이면 225개).
#   7) 프레임 점수       : 한 프레임이 들어간 클립들의 점수를 평균냅니다.
#
#   IoU(Intersection over Union): 두 상자가 겹친 넓이 ÷ 두 상자를 합친 넓이. 완전히 겹치면 1, 안 겹치면 0 입니다.
#
# 이 파일은 반드시 별도 프로세스로 실행합니다
#   FTCN 저장소에 utils, config, model 같은 이름의 폴더가 있어서, 프로젝트의 utils 폴더와 한 프로세스에 올리면
#   "import utils" 가 어느 쪽을 가리킬지 섞입니다. 그래서 이 파일은 프로젝트 코드를 import 하지 않고,
#   run_pilot.py 가 subprocess 로 따로 실행합니다.
#
# 공식 test_on_raw_video.py 와 달라진 점 (점수를 계산하는 방식은 그대로입니다)
#   - 없어진 NumPy 별칭(np.int, np.float)을 다시 연결합니다.
#   - torch 1.9 부터 Conv3d 생성자에 생긴 device/dtype 인자를 FTCN 의 합성곱 교체 코드에서 뺍니다.
#   - 얼굴 검출·랜드마크 가중치를 인터넷에서 받지 않고 받아둔 파일에서 읽습니다. 다른 주소를 받으려 하면 멈춥니다.
#   - FTCN 가중치는 모든 키와 모양이 일치할 때만 불러옵니다. 공식 코드는 일부만 맞아도 경고만 하고 넘어갑니다.
#   - cv2.copyMakeBorder 에 테두리 색을 dst 자리로 넘기는 부분을 value 로 바로잡습니다(OpenCV 4 이후 오류).
#   - 결과 영상(.avi)과 검출 캐시(.pth)를 입력 영상 옆에 쓰지 않습니다.
#   - GPU 의 TF32 근사 연산과 cudnn.benchmark 를 끕니다. 켜 두면 클립 묶음 크기에 따라 점수가 최대 0.076 달라졌습니다.
#   - 32프레임 클립을 여러 개씩 묶어 추론할 수 있습니다(--clip-batch). TF32 를 끈 상태에서 묶음 1과 4의
#     클립 점수 차이는 5e-7 이하였습니다(035 영상 전체, 2026-09-13).
#
# 결과 파일 (.npz, 영상마다 하나)에 들어 있는 값
#   n_frames, height, width : 프레임 수와 크기
#   faces         [프레임 수]     : 프레임마다 검출된 얼굴 수 (0 이면 얼굴을 못 찾음)
#   full_track    True/False     : 얼굴 추적이 영상 전체에서 끊기지 않았는지
#   segments      [구간 수, 2]    : 추적이 이어진 구간들의 [시작, 끝)
#   clip_scores   [클립 수]       : 32프레임 클립마다의 fake 확률
#   clip_frames   [클립 수, 32]   : 각 클립에 들어간 프레임 번호
#   clip_segments [클립 수]       : 각 클립이 속한 구간 번호
#   frame_scores  [프레임 수]     : 프레임 점수 (한 번도 채점되지 않은 프레임은 nan)
#   frame_hits    [프레임 수]     : 그 프레임이 들어간 클립 수
#   video_score                  : 모든 클립 점수의 평균 (공식 코드가 출력하는 영상 점수)
#   error         문자열          : 실패했으면 오류 내용, 성공이면 빈 문자열
#   elapsed_sec                  : 처리 시간(초)
#   .npz 는 numpy 배열 여러 개를 이름을 붙여 한 파일에 담는 형식입니다. np.load(경로)["frame_scores"] 처럼 꺼냅니다.
#
# 참고
#   - 공개된 FTCN 코드로 real 영상이 fake 로 나온다는 보고가 있습니다(GitHub 이슈 #5).
#     이 실행기로 Celeb-DF test 목록의 real 5개 / fake 5개를 채점했을 때 AUC 0.92 로 둘을 구분했습니다(2026-09-13).
#   - Conv3d 의 device 속성 오류(GitHub 이슈 #14)는 위의 device/dtype 처리로 해결됩니다.
#   - 속도: 256프레임 영상 하나에 약 35초 (RTX 5070 Laptop, --clip-batch 4).
#
# 실행 예 (프로젝트 폴더에서):
#   python models/ftcn_runner.py --out-dir D:/Graduation-project-data/processed/pilot/ftcn D:/some/video.mp4
#   python models/ftcn_runner.py --out-dir D:/Graduation-project-data/processed/pilot/ftcn --manifest D:/.../manifest.csv
# ============================================================================

import argparse
import csv
import os
import sys
import time
import traceback  # 오류가 난 위치와 내용을 문자열로 만들어 결과 파일에 남길 때 씁니다.
from pathlib import Path

import numpy as np
import yaml


# __file__는 현재 파일 경로입니다. 상위 두 단계인 프로젝트 폴더를 기준으로 설정 파일을 찾습니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# FTCN 본체 가중치 파일 이름
FTCN_WEIGHT = "ftcn_tt.pth"
# FTCN 코드가 torch.utils.model_zoo.load_url 로 인터넷에서 받으려는 파일들입니다. 이름이 같은 받아둔 파일로 대신합니다.
#   mobilenet0.25_Final.pth                      : 얼굴 검출(RetinaFace)
#   mobilenet_224_model_best_gdconv_external.pth : 68점 랜드마크
FACE_WEIGHTS = ("mobilenet0.25_Final.pth", "mobilenet_224_model_best_gdconv_external.pth")


# 여러 스크립트가 같은 경로와 기준값을 쓰도록 YAML 설정을 딕셔너리로 읽습니다.
def load_config():
    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return yaml.safe_load(file)


# 명령줄 옵션을 읽습니다. 기본 경로는 config.yaml 의 ftcn 항목에서 가져옵니다.
def parse_args():
    config = load_config()

    parser = argparse.ArgumentParser(
        description="FTCN 으로 영상의 프레임별 딥페이크 점수를 계산합니다."
    )
    # nargs="*" : 영상 경로를 0개 이상 받습니다. nargs="+" : 1개 이상 받습니다.
    parser.add_argument("videos", nargs="*", help="점수를 매길 영상 파일")
    parser.add_argument("--manifest", help="synthesize_transitions.py 가 만든 manifest.csv")
    parser.add_argument("--only", nargs="+", help="처리할 이름(clip_id 또는 파일 이름)만 고릅니다.")
    parser.add_argument("--out-dir", required=True, help="결과 .npz 를 저장할 폴더")
    parser.add_argument(
        "--ftcn-root",
        default=str(PROJECT_ROOT / config["ftcn"]["repo_dir"]),
        help="FTCN 공식 코드 폴더"
    )
    parser.add_argument(
        "--weights-dir",
        default=config["ftcn"]["weights_dir"],
        help="가중치 3개가 있는 폴더"
    )
    parser.add_argument("--clip-batch", type=int, default=1, help="한 번에 추론할 32프레임 클립 수")
    parser.add_argument("--detect-batch", type=int, default=16, help="한 번에 얼굴을 검출할 프레임 수")
    parser.add_argument("--max-frames", type=int, default=768, help="영상에서 읽을 최대 프레임 수 (공식 코드 768)")
    parser.add_argument("--overwrite", action="store_true", help="결과가 있어도 다시 계산합니다.")
    return parser.parse_args()


# 처리할 (이름, 영상 경로) 목록을 만듭니다.
# load_ftcn 이 작업 폴더를 FTCN 폴더로 바꾸기 때문에, 그 전에 모든 경로를 절대 경로(resolve)로 고정해 둡니다.
def collect_jobs(args):
    jobs = []

    # manifest 가 있으면 그 목록의 영상을 clip_id 이름으로 처리합니다.
    # clip_id 는 manifest 전체에서 유일하지 않습니다. FF++ 는 같은 번호의 영상을 기법마다 하나씩 갖기 때문에
    # Deepfakes/000_003 과 Face2Face/000_003 의 clip_id 가 같습니다. 그래서 source 열이 있으면
    # "source/clip_id" 를 이름으로 써서 결과 파일이 서로 덮어쓰지 않게 합니다(signals 폴더와 같은 구조).
    if args.manifest:
        with open(args.manifest, newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                name = f"{row['source']}/{row['clip_id']}" if row.get("source") else row["clip_id"]
                jobs.append((name, Path(row["path"]).resolve()))

    # 직접 준 영상 경로는 파일 이름(확장자 제외)으로 처리합니다.
    for video in args.videos:
        path = Path(video).resolve()
        jobs.append((path.stem, path))

    # --only 는 "source/clip_id" 전체와 clip_id 만 준 경우 둘 다 받습니다.
    if args.only:
        wanted = set(args.only)
        jobs = [job for job in jobs if job[0] in wanted or job[0].split("/")[-1] in wanted]

    return jobs


# NumPy 2 에서 없어진 별칭을 파이썬 기본 타입으로 다시 연결합니다.
# 예전 NumPy 의 np.int 는 파이썬 int 의 다른 이름일 뿐이었는데, FTCN 코드가 np.int 를 써서 지금은 오류가 납니다.
# np.__dict__ 는 numpy 모듈에 실제로 들어 있는 이름들입니다. 없을 때만 setattr 로 이름을 붙여 줍니다.
def install_numpy_aliases():
    for alias, builtin in (("int", int), ("float", float), ("bool", bool), ("object", object)):
        if alias not in np.__dict__:
            setattr(np, alias, builtin)


# 인터넷에서 가중치를 받는 함수를, 받아둔 파일을 읽는 함수로 바꿔 끼웁니다.
# 이렇게 실행 중에 라이브러리 함수를 다른 함수로 바꾸는 방식을 "몽키 패치(monkey patch)"라고 부릅니다.
# FTCN 코드는 torch.utils.model_zoo.load_url(주소) 를 부르므로, 그 이름이 가리키는 함수를 바꾸면 코드를 고치지 않고 동작을 바꿀 수 있습니다.
def install_offline_weights(torch, weights_dir):
    local = {name: weights_dir / name for name in FACE_WEIGHTS}
    missing = [str(path) for path in local.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"얼굴 검출/랜드마크 가중치가 없습니다: {missing}")

    # 함수 안의 함수입니다. 바깥 변수 local 을 기억한 채로 쓰입니다(클로저).
    # *args, **kwargs 는 원래 함수에 넘어오던 나머지 인자를 모두 받아서 무시하기 위한 것입니다.
    def load_local(url, *args, **kwargs):
        # 주소의 마지막 "/" 뒤가 파일 이름입니다.
        name = url.rsplit("/", 1)[-1]
        if name not in local:
            raise RuntimeError(f"FTCN 코드가 예상하지 못한 파일을 받으려 했습니다: {url}")
        # weights_only=True: 가중치 텐서만 읽고, 파일 안에 숨은 임의의 파이썬 코드는 실행하지 않는 안전한 방식입니다.
        return torch.load(local[name], map_location="cpu", weights_only=True)

    torch.utils.model_zoo.load_url = load_local
    torch.hub.load_state_dict_from_url = load_local


# FTCN 의 랜드마크 코드가 쓰는 cv2 를 감싸서, copyMakeBorder 의 인자 자리만 바로잡습니다.
# __getattr__ 은 "이 객체에 없는 속성을 찾을 때" 불리는 특별한 메서드입니다.
# 그래서 copyMakeBorder 를 뺀 나머지(cv2.resize 등)는 모두 진짜 cv2 로 그대로 넘어갑니다.
class OpenCvCompat:
    def __init__(self, cv2_module):
        self._cv2 = cv2_module

    def __getattr__(self, name):
        return getattr(self._cv2, name)

    def copyMakeBorder(self, src, top, bottom, left, right, border_type, *rest, **kwargs):
        # FTCN 은 테두리 색 0 을 7번째 위치 인자로 넘기는데, OpenCV 에서 그 자리는 출력 배열(dst)입니다.
        # 7번째 값이 배열이 아니면 테두리 색(value)으로 옮겨서 넘깁니다.
        if rest and not isinstance(rest[0], np.ndarray):
            kwargs.setdefault("value", rest[0])
            rest = rest[1:]
        return self._cv2.copyMakeBorder(src, top, bottom, left, right, border_type, *rest, **kwargs)


# FTCN 가중치를 모든 키와 모양이 맞을 때만 불러옵니다. 반환값은 불러온 텐서 수입니다.
# state_dict: {"층 이름": 가중치 텐서} 모양의 딕셔너리입니다. 예) "resnet.s1.pathway0_stem.conv.weight": 텐서
# 이름이나 모양이 하나라도 다르면, 그 층은 학습되지 않은 무작위 값으로 남아 점수가 조용히 틀어집니다.
def load_classifier_weights(torch, network, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    # 어떤 체크포인트는 {"state_dict": {...}} 처럼 한 번 더 감싸져 있어서 벗겨냅니다.
    if "state_dict" in state and not torch.is_tensor(state["state_dict"]):
        state = state["state_dict"]

    # 여러 GPU 로 학습한 모델은 키 앞에 "module." 이 붙어 있어 공식 코드와 같이 떼어냅니다.
    state = {
        (key[len("module."):] if key.startswith("module.") else key): value
        for key, value in state.items()
    }

    # 모델이 기대하는 키와 파일의 키를 집합 연산으로 비교합니다.
    #   set(A) - set(B) : A 에만 있는 것,  set(A) & set(B) : 둘 다 있는 것
    expected = network.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = sorted(
        key for key in set(state) & set(expected)
        if tuple(state[key].shape) != tuple(expected[key].shape)
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "FTCN 가중치가 모델 구조와 맞지 않습니다. "
            f"모델에만 있는 키 {len(missing)}개 {missing[:5]}, "
            f"가중치에만 있는 키 {len(unexpected)}개 {unexpected[:5]}, "
            f"모양이 다른 키 {len(mismatched)}개 {mismatched[:5]}"
        )

    # strict=True: 키가 완전히 일치하지 않으면 오류를 냅니다(위에서 이미 확인했지만 한 번 더 보장).
    network.load_state_dict(state, strict=True)
    return len(state)


# FTCN 코드를 불러오고 검출기, 랜드마크 모델, 분류기를 준비합니다.
# 순서가 중요합니다: (1) 호환성 패치 → (2) FTCN 폴더로 이동 → (3) 설정 읽기 → (4) 모델 만들기 → (5) GPU 설정
def load_ftcn(ftcn_root, weights_dir):
    install_numpy_aliases()

    # torch 를 함수 안에서 import 합니다. 결과가 이미 다 있어서 할 일이 없을 때는 무거운 torch 를 불러오지 않기 위해서입니다.
    import torch
    import torch.hub
    import torch.utils.model_zoo

    # FTCN 코드가 가중치를 받으러 가기 "전에" 바꿔 끼워야 합니다.
    install_offline_weights(torch, weights_dir)

    # FTCN 은 자기 폴더를 기준으로 import 하고 설정 파일을 찾습니다.
    #   os.chdir     : 작업 폴더를 FTCN 폴더로 바꿈
    #   sys.path.insert(0, ...) : "import config" 같은 이름을 FTCN 폴더에서 가장 먼저 찾게 함
    os.chdir(ftcn_root)
    sys.path.insert(0, str(ftcn_root))

    import cv2
    from config import config as cfg

    # FTCN 설정: 기본 설정(root_setting.yaml) 위에 FTCN 전용 설정(setting/ftcn_tt.yaml)을 덮어씁니다.
    # freeze() 는 이후에 실수로 설정을 바꾸지 못하게 잠급니다. 클립 길이(32)와 입력 크기(224)가 여기서 정해집니다.
    cfg.init_with_yaml()
    cfg.update_with_yaml("ftcn_tt.yaml")
    cfg.freeze()

    import test_tools.ct.face_alignment.predictor as landmark_module
    landmark_module.cv2 = OpenCvCompat(cv2)

    # 이 import 가 실행될 때 FTCN 코드가 얼굴 검출기와 랜드마크 모델을 만듭니다(GPU 0).
    from test_tools.common import detector, get_lm68
    from test_tools.ct.detection.utils import get_valid_faces
    from test_tools.ct.operations import find_longest, multiple_tracking
    from test_tools.faster_crop_align_xray import FasterCropAlignXRay
    from test_tools.utils import flatten, get_crop_box, partition
    from utils.plugin_loader import PluginLoader

    # 설정에 적힌 분류기 이름(i3d_temporal_var_fix_dropout_tt_cfg)으로 모델 클래스를 찾습니다.
    classifier_class = PluginLoader.get_classifier(cfg.classifier_type)
    # 클래스가 정의된 파이썬 모듈(파일) 객체를 꺼냅니다. 그 파일의 전역 변수 parameters 를 고치기 위해서입니다.
    classifier_module = sys.modules[classifier_class.__module__]
    # FTCN 은 Conv3d 생성자의 인자 이름을 모두 읽어 기존 합성곱에서 같은 이름의 값을 가져옵니다.
    # torch 1.9 부터 생긴 device/dtype 은 합성곱 객체에 그런 속성이 없어서 목록에서 뺍니다.
    classifier_module.parameters = [
        name for name in classifier_module.parameters if name not in ("device", "dtype")
    ]

    classifier = classifier_class()
    loaded = load_classifier_weights(torch, classifier.network, weights_dir / FTCN_WEIGHT)
    # .cuda(): 모델을 GPU 메모리로 옮김.  .eval(): 평가 모드(Dropout 끄고, BatchNorm 은 학습 때 저장한 통계 사용)
    classifier.cuda()
    classifier.eval()

    # PyTorch 는 RTX 30 시리즈 이후 GPU 의 합성곱에 TF32 근사 연산을 기본으로 씁니다.
    # 이 모델에서는 그 오차 때문에 클립을 묶는 개수에 따라 점수가 달라졌습니다
    # (035 영상 전체에서 클립 점수 최대 0.076 차이, 끄면 1.8e-7 수준이고 CPU 계산과 일치).
    # cudnn.benchmark 는 입력 크기마다 가장 빠른 계산법을 골라 쓰는 기능인데, 고른 방법에 따라 결과가 미세하게 달라질 수 있습니다.
    # FTCN 의 얼굴 검출기 코드가 cudnn.benchmark 를 켜므로 모델을 모두 만든 뒤에 끕니다.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    print(f"FTCN 가중치 텐서 {loaded}개를 모두 불러왔습니다.")

    # score_video 에서 쓸 것들을 딕셔너리 하나로 묶어 넘깁니다.
    return {
        "torch": torch,
        "cfg": cfg,
        "detector": detector,
        "get_lm68": get_lm68,
        "get_valid_faces": get_valid_faces,
        "multiple_tracking": multiple_tracking,
        "find_longest": find_longest,
        "crop_align": FasterCropAlignXRay(cfg.imsize),
        "flatten": flatten,
        "partition": partition,
        "get_crop_box": get_crop_box,
        "classifier": classifier,
    }


# 공식 코드와 같이 OpenCV 로 프레임을 읽고 BGR 을 RGB 로 뒤집습니다. 검출기에도 이 RGB 프레임이 들어갑니다.
# OpenCV 는 색을 B, G, R 순서로 저장합니다. frame[..., ::-1] 은 마지막 축(색 채널)의 순서를 거꾸로 해서 R, G, B 로 만듭니다.
#   "..." 는 앞의 축들을 모두 그대로 둔다는 뜻입니다.
def read_rgb_frames(path, max_frames):
    import cv2

    capture = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < max_frames:
        # read() 는 (성공 여부, 프레임) 을 돌려줍니다. 영상 끝이면 성공 여부가 False 입니다.
        ok, frame = capture.read()
        if not ok:
            break
        # 거꾸로 뒤집은 배열은 메모리 순서가 뒤집혀 있어서 ascontiguousarray 로 정리해 둡니다.
        frames.append(np.ascontiguousarray(frame[..., ::-1]))
    capture.release()
    return frames


# 공식 코드의 클립 뽑기를 그대로 옮겼습니다.
# super_clip_sizes: 추적이 이어진 구간마다의 길이. 예) 끊김 없이 256프레임이면 [256], 128에서 끊기면 [128, 128]
# 구간 안에서 연속 32프레임을 1프레임씩 밀면서 클립을 만듭니다.
#   예) 길이 256 → 시작 위치 0, 1, ..., 224 → 클립 225개
# 32프레임보다 짧은 구간은 앞뒤를 거울처럼 이어 붙여 채웁니다.
#   예) 길이 4 인 [0, 1, 2, 3] → ... 2, 1, 0, 1, 2, 3, 2, 1, 0 ... 처럼 왕복하며 늘립니다.
# 반환값: ([(구간 번호, 구간 안 순번), ...] 32개짜리 클립 목록, 클립마다의 구간 번호)
def sample_clips(super_clip_sizes, clip_size):
    pad_length = clip_size - 1
    clips = []
    clip_segments = []

    for super_index, size in enumerate(super_clip_sizes):
        inner = list(range(size))
        if size < clip_size:
            # [1:-1][::-1] = 양 끝을 뺀 나머지를 거꾸로. 이걸 붙이면 0,1,2,3 → 2,1,0,1,2,3 처럼 왕복이 됩니다.
            post = inner[1:-1][::-1] + inner
            post_length = len(post)
            post = (post * (pad_length // post_length + 1))[:pad_length]
            pre = inner + inner[1:-1][::-1]
            # 공식 코드는 여기서도 post 의 길이를 씁니다(GitHub 이슈 #6 에서 지적된 부분). 결과를 맞추려고 그대로 둡니다.
            pre = (pre * (pad_length // post_length + 1))[-pad_length:]
            inner = pre + inner + post

        for start in range(len(inner) - clip_size + 1):
            clips.append([(super_index, t) for t in inner[start:start + clip_size]])
            clip_segments.append(super_index)

    return clips, clip_segments


# 영상 하나의 점수를 계산합니다. 반환값은 .npz 로 저장할 배열 딕셔너리입니다.
def score_video(path, ftcn, clip_batch, detect_batch, max_frames):
    torch = ftcn["torch"]

    frames = read_rgb_frames(path, max_frames)
    frame_count = len(frames)
    if frame_count == 0:
        raise RuntimeError(f"프레임을 읽지 못했습니다: {path}")
    # shape[:2] = (세로, 가로)
    height, width = frames[0].shape[:2]

    # 1. 얼굴 검출. 공식 코드와 같은 신뢰도 기준 0.5 를 씁니다.
    #    partition 으로 프레임을 detect_batch(16)장씩 나눠 GPU 에 넣고, flatten 으로 결과를 다시 한 줄로 이어 붙입니다.
    #    detections[프레임 번호] = [(얼굴 상자, 5점, 신뢰도), ...]  ← 프레임마다 얼굴 목록
    raw = ftcn["flatten"]([
        ftcn["detector"].detect(chunk) for chunk in ftcn["partition"](frames, detect_batch)
    ])
    detections = ftcn["get_valid_faces"](raw, thres=0.5)
    faces = np.array([len(frame_faces) for frame_faces in detections], dtype=np.int16)

    # 2. 얼굴마다 68점 랜드마크를 구해서, 얼굴 정보를 (상자, 5점, 68점, 신뢰도) 로 늘립니다.
    landmarks = ftcn["get_lm68"](frames, detections)
    detections = [
        [(box, lm5, lm68, score) for (box, lm5, score), lm68 in zip(frame_faces, frame_landmarks)]
        for frame_faces, frame_landmarks in zip(detections, landmarks)
    ]

    # 3. 추적. 영상 전체에서 이어지는 얼굴이 있으면 그 얼굴들을, 없으면 이어지는 구간들로 나눠서 씁니다.
    #    tracks[i] = i번째 얼굴을 프레임마다 모은 목록, segments[i] = 그 얼굴이 이어진 [시작, 끝)
    tracks = ftcn["multiple_tracking"](detections)
    full_track = len(tracks) > 0
    if full_track:
        segments = [(0, frame_count)] * len(tracks)
    else:
        segments, tracks = ftcn["find_longest"](detections)

    # 프레임마다 "얼굴을 넉넉히 둘러싼 크롭 이미지"와 좌표 정보를 저장해 둡니다. 클립을 만들 때 꺼내 씁니다.
    #   키: (구간 번호, 구간 안 순번),  값: (크롭 이미지, 좌표 정보, 원래 프레임 번호)
    storage = {}
    super_clip_sizes = []
    for track_index, ((start, end), track) in enumerate(zip(segments, tracks)):
        assert len(track) == end - start
        super_clip_sizes.append(len(track))
        for j, (face, frame_index) in enumerate(zip(track, range(start, end))):
            box, lm5, lm68 = face[:3]
            # 얼굴 상자를 가로·세로 각각 50% 씩 넓힌 상자(scale=0.5)로 크롭합니다.
            big_box = ftcn["get_crop_box"]((height, width), box, scale=0.5)
            # 크롭 이미지 기준 좌표로 바꾸려고 크롭 상자의 왼쪽 위 좌표를 뺍니다.
            #   [None, :] 는 배열에 축을 하나 추가해 [2] 모양을 [1, 2] 로 만들어, 점 여러 개에서 한꺼번에 뺄 수 있게 합니다.
            top_left = big_box[:2][None, :]
            info = (
                (box.reshape(2, 2) - top_left).reshape(-1),
                lm5 - top_left,
                lm68 - top_left,
                big_box,
            )
            x1, y1, x2, y2 = big_box
            # 이미지 배열은 [세로, 가로] 순서라 [y1:y2, x1:x2] 로 자릅니다.
            storage[(track_index, j)] = (frames[frame_index][y1:y2, x1:x2], info, frame_index)

    # 4. 32프레임 클립마다 점수
    clip_size = ftcn["cfg"].clip_size
    clips, clip_segments = sample_clips(super_clip_sizes, clip_size)

    # ImageNet 평균과 표준편차(0~255 기준)로 입력을 정규화합니다. 모델이 학습 때 본 입력 분포와 맞추기 위해서입니다.
    # view(1, 3, 1, 1, 1): [묶음, 채널, 시간, 세로, 가로] 5차원 입력의 채널 축에 맞춰 모양을 바꿉니다.
    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255], device="cuda").view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255], device="cuda").view(1, 3, 1, 1, 1)

    clip_scores = np.empty(len(clips), dtype=np.float32)
    clip_frames = np.empty((len(clips), clip_size), dtype=np.int32)

    # 클립을 clip_batch 개씩 묶어서 GPU 에 한 번에 넣습니다.
    for batch_start in range(0, len(clips), clip_batch):
        batch = clips[batch_start:batch_start + clip_batch]
        tensors = []
        for offset, clip in enumerate(batch):
            images = [storage[key][0] for key in clip]
            infos = [storage[key][1] for key in clip]
            clip_frames[batch_start + offset] = [storage[key][2] for key in clip]
            # 클립 32프레임의 5점 좌표로 변환 하나를 구해, 모든 프레임을 같은 방식으로 224x224 에 맞춥니다.
            # aligned 모양: [32, 224, 224, 3]
            _, aligned = ftcn["crop_align"](infos, images)
            # permute(3, 0, 1, 2): [시간, 세로, 가로, 채널] → [채널, 시간, 세로, 가로] (PyTorch 3D 합성곱의 입력 순서)
            tensors.append(torch.as_tensor(aligned, dtype=torch.float32).permute(3, 0, 1, 2))

        # torch.stack 으로 묶음 축을 앞에 붙여 [묶음, 3, 32, 224, 224] 를 만들고, GPU 로 옮겨 정규화합니다.
        inputs = torch.stack(tensors).cuda().sub(mean).div(std)
        # no_grad: 학습이 아니라서 기울기(gradient)를 계산하지 않습니다. 메모리와 시간을 아낍니다.
        with torch.no_grad():
            output = ftcn["classifier"](inputs)["final_output"]
        # 모델 출력 [묶음, 1] 을 [묶음] 으로 펴서 CPU 의 numpy 배열로 가져옵니다. 값은 fake 확률(0~1)입니다.
        clip_scores[batch_start:batch_start + len(batch)] = (
            output.float().reshape(len(batch), -1)[:, 0].cpu().numpy()
        )

    # 5. 프레임 점수 = 그 프레임이 들어간 클립 점수의 평균. 거울 채움으로 두 번 들어가면 두 번 셉니다(공식 코드와 같음).
    sums = np.zeros(frame_count, dtype=np.float64)
    hits = np.zeros(frame_count, dtype=np.int32)
    for score, frame_ids in zip(clip_scores, clip_frames):
        for frame_id in frame_ids:
            sums[frame_id] += score
            hits[frame_id] += 1
    # 한 번도 클립에 안 들어간 프레임(얼굴을 못 찾은 프레임 등)은 nan(값 없음)으로 둡니다.
    frame_scores = np.full(frame_count, np.nan, dtype=np.float32)
    # hits > 0 은 True/False 배열이고, 이걸로 인덱싱하면 True 인 위치만 골라 계산합니다(불리언 인덱싱).
    frame_scores[hits > 0] = (sums[hits > 0] / hits[hits > 0]).astype(np.float32)

    return {
        "n_frames": np.int32(frame_count),
        "height": np.int32(height),
        "width": np.int32(width),
        "faces": faces,
        "full_track": np.bool_(full_track),
        "segments": np.array(segments, dtype=np.int32).reshape(-1, 2),
        "clip_scores": clip_scores,
        "clip_frames": clip_frames,
        "clip_segments": np.array(clip_segments, dtype=np.int16),
        "frame_scores": frame_scores,
        "frame_hits": hits.astype(np.int16),
        "video_score": np.float32(clip_scores.mean() if len(clip_scores) else np.nan),
    }


def main():
    args = parse_args()

    ftcn_root = Path(args.ftcn_root).resolve()
    weights_dir = Path(args.weights_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 이미 결과(.npz)가 있는 영상은 건너뜁니다. 그래서 중간에 끊겨도 다시 실행하면 남은 것만 처리합니다.
    jobs = collect_jobs(args)
    pending = [
        (name, path) for name, path in jobs
        if args.overwrite or not (out_dir / f"{name}.npz").exists()
    ]
    print(f"영상 {len(jobs)}개 중 {len(pending)}개를 처리합니다 (나머지는 결과가 이미 있음).")
    if not pending:
        return

    # 모델은 한 번만 불러오고 모든 영상에 재사용합니다(불러오는 데 시간이 걸리기 때문).
    ftcn = load_ftcn(ftcn_root, weights_dir)

    # enumerate(pending, 1): 번호를 1부터 셉니다.
    for index, (name, path) in enumerate(pending, 1):
        started = time.time()
        # 영상 하나가 실패해도 전체가 멈추지 않도록 오류를 잡아서 결과 파일에 기록하고 다음 영상으로 넘어갑니다.
        try:
            result = score_video(path, ftcn, args.clip_batch, args.detect_batch, args.max_frames)
            result["error"] = np.array("")
        except Exception:
            result = {"error": np.array(traceback.format_exc(limit=8))}
        elapsed = time.time() - started
        result["elapsed_sec"] = np.float32(elapsed)
        # **result: 딕셔너리의 각 키를 이름으로 붙여 배열들을 한 파일에 압축 저장합니다.
        # name 에 "source/" 가 붙어 있으면 하위 폴더를 만들어 그 안에 저장합니다.
        out_path = out_dir / f"{name}.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **result)

        if str(result["error"]):
            print(f"[{index}/{len(pending)}] {name} 실패 ({elapsed:.1f}초)\n{result['error']}")
            continue

        no_face = int((result["faces"] == 0).sum())
        scored = int((result["frame_hits"] > 0).sum())
        track_note = "끊김 없음" if result["full_track"] else f"끊겨서 구간 {len(result['segments'])}개"
        print(
            f"[{index}/{len(pending)}] {name} | {int(result['n_frames'])}프레임 | "
            f"얼굴 없는 프레임 {no_face} | 추적 {track_note} | 채점된 프레임 {scored} | "
            f"클립 {len(result['clip_scores'])}개 | 영상 점수 {float(result['video_score']):.4f} | {elapsed:.1f}초"
        )


# 이 파일을 직접 실행할 때만 main() 을 부릅니다. 다른 파일에서 import 할 때는 실행되지 않습니다.
if __name__ == "__main__":
    main()
