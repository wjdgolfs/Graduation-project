# ============================================================================
# FTCN 실행기의 결과 파일 이름 테스트
# ----------------------------------------------------------------------------
# 왜 필요한가
#   FF++ 는 같은 번호의 영상을 조작 기법마다 하나씩 갖습니다(Deepfakes/000_003, Face2Face/000_003 ...).
#   그래서 clip_id 만으로 결과 .npz 이름을 지으면 기법이 다른 클립이 서로를 덮어씁니다.
#   실제로 벤치마크 210클립을 채점했을 때 파일이 133개만 남았습니다(2026-10-02).
#   이 테스트는 그 상황을 작은 manifest 로 재현해 이름이 겹치지 않는지 확인합니다.
# 실행: .venv/Scripts/python.exe -m pytest tests
# ============================================================================

import csv
import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.ftcn_runner import collect_jobs  # noqa: E402


# 테스트용 manifest 를 만듭니다. columns 에 없는 열은 쓰지 않습니다.
def write_manifest(path, rows, columns):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


# source 열이 있으면 "source/clip_id" 가 되어 기법이 달라도 이름이 겹치지 않아야 합니다.
def test_names_are_unique_across_sources(tmp_path):
    rows = [
        {"clip_id": "000_003__control", "source": source, "path": f"D:/clips/{source}/000_003__control.mp4"}
        for source in ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
    ]
    rows.append({"clip_id": "000__control", "source": "original", "path": "D:/clips/original/000__control.mp4"})
    manifest = write_manifest(tmp_path / "manifest.csv", rows, ["clip_id", "source", "path"])

    jobs = collect_jobs(SimpleNamespace(manifest=str(manifest), videos=[], only=None))
    names = [name for name, _ in jobs]

    assert len(names) == len(set(names)) == 5
    assert "Deepfakes/000_003__control" in names
    assert "NeuralTextures/000_003__control" in names
    assert "original/000__control" in names


# source 열이 없는 manifest(0단계 파일럿)는 예전처럼 clip_id 를 그대로 씁니다.
def test_manifest_without_source_keeps_clip_id(tmp_path):
    rows = [{"clip_id": "035__dissolve64", "path": "D:/pilot/clips/035__dissolve64.mp4"}]
    manifest = write_manifest(tmp_path / "manifest.csv", rows, ["clip_id", "path"])

    jobs = collect_jobs(SimpleNamespace(manifest=str(manifest), videos=[], only=None))

    assert [name for name, _ in jobs] == ["035__dissolve64"]


# --only 는 "source/clip_id" 전체와 clip_id 만 준 경우를 모두 받아야 합니다.
def test_only_accepts_both_forms(tmp_path):
    rows = [
        {"clip_id": "000_003__control", "source": "Deepfakes", "path": "D:/a.mp4"},
        {"clip_id": "000_003__control", "source": "Face2Face", "path": "D:/b.mp4"},
        {"clip_id": "000__control", "source": "original", "path": "D:/c.mp4"},
    ]
    manifest = write_manifest(tmp_path / "manifest.csv", rows, ["clip_id", "source", "path"])

    full = collect_jobs(SimpleNamespace(manifest=str(manifest), videos=[], only=["Face2Face/000_003__control"]))
    assert [name for name, _ in full] == ["Face2Face/000_003__control"]

    short = collect_jobs(SimpleNamespace(manifest=str(manifest), videos=[], only=["000_003__control"]))
    assert len(short) == 2
