# ============================================================================
# 3단계: 장면 전환 벤치마크 만들기
# ----------------------------------------------------------------------------
# 무엇을 하나
#   FF++ 테스트 분할 영상에 장면 전환(하드컷/페이드/디졸브)을 합성해 평가용 데이터셋을 만듭니다.
#   0단계 파일럿(preprocessing/synthesize_transitions.py)은 real 영상만 다뤘지만, 여기서는
#   real 과 fake 를 모두 기준 영상으로 씁니다. 그래야 전환이
#     - real 을 fake 로 오판하게 만드는지 (오탐)
#     - fake 의 탐지를 방해하는지 (미탐)
#   양쪽을 모두 잴 수 있습니다.
#
# 전환 상대(partner) 규칙
#   real 은 다른 real 과, fake 는 같은 기법의 다른 fake 와 이어 붙입니다.
#   그래야 전환을 넣어도 그 영상의 정답(real/fake)이 바뀌지 않습니다.
#
# 영상 하나가 만드는 클립 9개
#   control(전환 없음) + hard_cut + fade 8/16/32 + dissolve 8/16/32/64
#   전환 구간 밖의 프레임은 control 과 완전히 같으므로, 점수 차이는 전환 때문이라고 말할 수 있습니다.
#   (프레임 번호 규칙과 전환 계산식은 preprocessing/synthesize_transitions.py 머리말에 설명해 두었습니다.)
#
# 입력: FF++ c23 영상(원본 + 조작 4종)과 공식 분할 json
# 출력: benchmark.output_dir/clips/<기법>/<영상>__<전환>.mp4
#       benchmark.output_dir/manifest.csv   클립마다 정답·전환 종류·전환 구간을 적은 목록
#
# 실행 예 (프로젝트 폴더에서)
#   python preprocessing/build_transition_benchmark.py --real-videos 2 --fake-videos 4   시험
#   python preprocessing/build_transition_benchmark.py                                   설정대로 전체
#   python preprocessing/build_transition_benchmark.py --overwrite                       다시 만들기
# ============================================================================

import argparse
import csv
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml
from tqdm import tqdm


# __file__는 현재 파일 경로입니다. 상위 두 단계인 프로젝트 폴더를 기준으로 설정 파일을 찾습니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# 병렬 처리용 자식 프로세스도 이 파일을 다시 읽으므로 같은 경로가 들어갑니다.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 전환 계산은 파일럿 코드의 함수를 그대로 씁니다(이미 tests/test_transitions.py 로 검증된 코드).
from preprocessing.synthesize_transitions import (  # noqa: E402
    FFPP_DIR,
    FFPP_ORIGINAL_DIR,
    assign_partners,
    compose_transition,
    load_split_ids,
    match_resolution,
    transition_window,
)
from utils.video_io import probe_video, read_frames, write_frames  # noqa: E402


MANIFEST_COLUMNS = [
    "clip_id", "source", "label", "kind", "length", "base_id", "partner_id",
    "window_start", "window_end", "frames", "fps", "width", "height", "resized", "fps_mismatch", "path",
]


# 여러 스크립트가 같은 경로와 기준값을 쓰도록 YAML 설정을 딕셔너리로 읽습니다.
def load_config():
    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return yaml.safe_load(file)


# 기법별 영상 폴더. original 은 원본, 나머지는 조작 영상입니다.
def source_dir(raw, source):
    if source == "original":
        return Path(raw) / FFPP_ORIGINAL_DIR
    return Path(raw) / FFPP_DIR / "manipulated_sequences" / source / "c23" / "videos"


# 클립 파일 이름. 길이가 있는 전환만 숫자를 붙입니다.
#   예) 035__control.mp4, 035__hard_cut.mp4, 035__dissolve32.mp4
def clip_name(base_id, kind, length):
    return f"{base_id}__{kind}{length}.mp4" if length else f"{base_id}__{kind}.mp4"


# 분할에 속하고 충분히 긴 영상을 이름순으로 count 개 고릅니다.
# 조작 영상 "A_B" 는 A 와 B 가 모두 그 분할에 있어야 합니다(FF++ 는 쌍 단위로 분할을 나눕니다).
def select_videos(folder, split_ids, count, segment):
    split = set(split_ids)
    chosen = {}
    for path in sorted(folder.glob("*.mp4")):
        if not all(part in split for part in path.stem.split("_")):
            continue
        info = probe_video(path)
        # segment 장보다 짧은 영상은 앞에서 그만큼 잘라낼 수 없으므로 뺍니다.
        if info["frames"] and info["frames"] < segment:
            continue
        chosen[path.stem] = {**info, "path": path}
        if len(chosen) == count:
            break
    return chosen


# 기준 영상 하나가 만들 클립 9개의 계획(정답·전환 구간·저장 경로)을 세웁니다. 파일을 만들지는 않습니다.
def plan_clips(base_id, source, label, infos, partner_id, settings, clip_root):
    base, partner = infos[base_id], infos[partner_id]
    conditions = [("control", 0)] + [
        (kind, length) for kind, lengths in settings["transitions"].items() for length in lengths
    ]

    plans = []
    for kind, length in conditions:
        start, end = (0, 0) if kind == "control" else transition_window(kind, length, settings["transition_center"])
        name = clip_name(base_id, kind, length)
        plans.append({
            "clip_id": name[:-4],
            "source": source,
            "label": label,
            "kind": kind,
            "length": length,
            "base_id": base_id,
            "partner_id": "" if kind == "control" else partner_id,
            "window_start": start,
            "window_end": end,
            "frames": settings["segment_frames"],
            "fps": round(base["fps"], 3),
            "width": base["width"],
            "height": base["height"],
            # 상대 영상의 크기나 fps 가 달랐는지 기록해 둡니다(결과 해석 때 확인용).
            "resized": int((partner["height"], partner["width"]) != (base["height"], base["width"])),
            "fps_mismatch": int(abs(partner["fps"] - base["fps"]) > 0.01),
            "path": str(Path(clip_root) / source / name),
        })
    return plans


# 자식 프로세스가 맡는 일: 기준 영상 하나의 클립들을 만듭니다.
# 원본 두 개를 한 번만 디코딩해서 클립 9개를 모두 만듭니다(디코딩이 가장 오래 걸리기 때문).
def build_base(job):
    spec, plans, overwrite = job
    todo = [plan for plan in plans if overwrite or not Path(plan["path"]).exists()]
    if not todo:
        return {"base_id": spec["base_id"], "made": 0, "error": "", "seconds": 0.0}

    started = time.time()
    try:
        Path(todo[0]["path"]).parent.mkdir(parents=True, exist_ok=True)
        segment = spec["segment_frames"]
        base, base_fps = read_frames(spec["base_path"], start=0, count=segment)
        partner, _ = read_frames(spec["partner_path"], start=0, count=segment)
        if len(base) < segment or len(partner) < segment:
            raise RuntimeError(f"프레임 부족: 기준 {len(base)}장, 상대 {len(partner)}장 (필요 {segment}장)")
        # 상대 영상의 해상도가 다르면 기준 영상에 맞춥니다(전환은 같은 크기끼리만 섞을 수 있습니다).
        partner, _ = match_resolution(partner, base.shape[1], base.shape[2])

        for plan in todo:
            frames = base if plan["kind"] == "control" else compose_transition(
                base, partner, plan["kind"], plan["length"], spec["transition_center"])
            write_frames(plan["path"], frames, fps=base_fps, crf=spec["crf"], preset=spec["preset"])
        return {"base_id": spec["base_id"], "made": len(todo), "error": "", "seconds": round(time.time() - started, 1)}
    except Exception:
        return {"base_id": spec["base_id"], "made": 0, "error": traceback.format_exc(limit=3).splitlines()[-1], "seconds": 0.0}


def main():
    parser = argparse.ArgumentParser(description="FF++ 영상에 장면 전환을 합성해 3단계 벤치마크를 만듭니다.")
    parser.add_argument("--real-videos", type=int, default=None, help="기준으로 쓸 원본(real) 영상 수")
    parser.add_argument("--fake-videos", type=int, default=None, help="기준으로 쓸 조작(fake) 영상 수. 기법 4종에 고르게 나눕니다.")
    parser.add_argument("--workers", type=int, default=None, help="동시에 돌릴 프로세스 수 (기본: 설정값)")
    parser.add_argument("--overwrite", action="store_true", help="이미 만든 클립도 다시 만듭니다.")
    args = parser.parse_args()

    config = load_config()
    settings = config["benchmark"]
    raw = config["paths"]["raw"]
    workers = args.workers or settings["workers"]
    real_count = args.real_videos if args.real_videos is not None else settings["real_videos"]
    fake_count = args.fake_videos if args.fake_videos is not None else settings["fake_videos"]
    clip_root = Path(settings["output_dir"]) / "clips"

    split_ids = load_split_ids(raw, settings["split"])
    methods = settings["methods"]
    # 기법마다 몇 개씩 고를지 나눕니다. 딱 나눠떨어지지 않으면 앞쪽 기법이 하나씩 더 맡습니다.
    per_method = [fake_count // len(methods) + (1 if index < fake_count % len(methods) else 0)
                  for index in range(len(methods))]

    groups = [("original", 0, real_count)] + [(method, 1, count) for method, count in zip(methods, per_method)]
    specs, all_plans = [], []
    for source, label, count in groups:
        if count <= 0:
            continue
        infos = select_videos(source_dir(raw, source), split_ids, count, settings["segment_frames"])
        if len(infos) < 2:
            print(f"건너뜀: {source} (쓸 수 있는 영상 {len(infos)}개, 전환 상대가 없습니다)")
            continue
        partners = assign_partners(infos)
        print(f"{source:<16} 영상 {len(infos)}개 (요청 {count}개)")
        for base_id in sorted(infos):
            plans = plan_clips(base_id, source, label, infos, partners[base_id], settings, clip_root)
            all_plans.extend(plans)
            specs.append(({
                "base_id": f"{source}/{base_id}",
                "base_path": str(infos[base_id]["path"]),
                "partner_path": str(infos[partners[base_id]]["path"]),
                "segment_frames": settings["segment_frames"],
                "transition_center": settings["transition_center"],
                "crf": settings["crf"],
                "preset": settings["preset"],
            }, plans))

    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "manifest.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(all_plans)
    print(f"\n기준 영상 {len(specs)}개 → 클립 {len(all_plans)}개 | 목록: {output_dir / 'manifest.csv'}")

    todo = sum(1 for plan in all_plans if args.overwrite or not Path(plan["path"]).exists())
    print(f"만들 클립 {todo}개 (이미 있는 것은 건너뜁니다), 프로세스 {workers}개")
    if not todo:
        return

    started = time.time()
    logs = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(build_base, (spec, plans, args.overwrite)) for spec, plans in specs]
        for future in tqdm(as_completed(futures), total=len(futures), unit="영상"):
            logs.append(future.result())

    failed = [log for log in logs if log["error"]]
    made = sum(log["made"] for log in logs)
    elapsed = (time.time() - started) / 60
    print(f"클립 {made}개 생성 / 실패한 기준 영상 {len(failed)}개 | 전체 {elapsed:.1f}분")
    for log in failed[:5]:
        print(f"  실패: {log['base_id']} - {log['error']}")


# 병렬 처리(자식 프로세스)가 있는 코드는 윈도우에서 반드시 이 조건 안에서 시작해야 합니다.
if __name__ == "__main__":
    main()
