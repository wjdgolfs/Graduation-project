# ============================================================================
# FF++ 병렬 다운로드 테스트 (인터넷 없이 확인)
# ----------------------------------------------------------------------------
# 이 컴퓨터 안(127.0.0.1)에 작은 웹 서버를 띄워 확인합니다.
#   - 정상 파일은 끝까지 받아 진짜 이름으로 저장하고, 받은 바이트 수를 계수기에 더해야 합니다.
#   - 서버가 알려 준 길이보다 적게 보내고 연결을 끊으면(IncompleteRead) 실패로 처리하고 .part 를 남기지 않아야 합니다.
#   - 이미 있는 파일은 건너뛰고, 실패한 파일은 목록으로 돌려줘야 합니다.
# 실행: .venv/Scripts/python.exe -m pytest tests
# ============================================================================

import importlib.util
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# scripts 폴더는 패키지(__init__.py)가 아니라서, 파일 경로로 직접 불러옵니다.
_spec = importlib.util.spec_from_file_location("download_ffpp_batch", PROJECT_ROOT / "scripts" / "download_ffpp_batch.py")
download = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(download)

BODY = bytes(range(256)) * 1000  # 256,000 바이트


# 테스트용 웹 서버가 주소마다 어떻게 응답할지 정합니다.
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/ok.mp4", "/short.mp4"):
            self.send_error(404)
            return
        self.send_response(200)
        # 두 경우 모두 "전체 길이를 보내겠다"고 알립니다.
        self.send_header("Content-Length", str(len(BODY)))
        self.end_headers()
        # short.mp4 는 절반만 보내고 연결을 끊습니다(서버가 중간에 끊는 상황 흉내).
        self.wfile.write(BODY if self.path == "/ok.mp4" else BODY[: len(BODY) // 2])

    # 테스트 출력이 지저분해지지 않게 서버 접속 기록을 끕니다.
    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    # 포트 0: 비어 있는 포트를 운영체제가 골라 줍니다.
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/"
    httpd.shutdown()
    httpd.server_close()


def test_download_one_saves_complete_file_and_counts_bytes(server, tmp_path):
    counter = download.ByteCounter()
    out_path = tmp_path / "ok.mp4"
    assert download.download_one(server + "ok.mp4", out_path, counter, retries=1) == "ok"
    assert out_path.read_bytes() == BODY
    assert counter.total == len(BODY)
    assert download.download_one(server + "ok.mp4", out_path, counter, retries=1) == "skip"


def test_download_one_rejects_truncated_file(server, tmp_path):
    out_path = tmp_path / "short.mp4"
    assert download.download_one(server + "short.mp4", out_path, download.ByteCounter(), retries=1) == "fail"
    assert not out_path.exists()
    assert not (tmp_path / "short.mp4.part").exists()


def test_download_parallel_skips_existing_and_lists_failures(server, tmp_path, monkeypatch):
    # 재시도 사이 대기(5초, 10초)를 없애 테스트를 빠르게 합니다.
    monkeypatch.setattr(download.time, "sleep", lambda seconds: None)
    (tmp_path / "ok.mp4").write_bytes(b"already here")
    counts, failed = download.download_parallel(["ok.mp4", "short.mp4", "missing.mp4"], server, tmp_path, workers=2)
    assert counts == {"ok": 0, "skip": 1, "fail": 2}
    assert sorted(failed) == ["missing.mp4", "short.mp4"]
    assert not list(tmp_path.glob("*.part"))


def test_human_size():
    assert download.human_size(512) == "512B"
    assert download.human_size(87040) == "85.0KB"
    assert download.human_size(1288490188) == "1.20GB"
