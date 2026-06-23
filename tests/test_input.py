# -*- coding: utf-8 -*-
"""analyze_face image 인자 입력 분기 — 로컬 경로 / data-URI / base64 수용 + 실패 가드.

헤르메스 게이트웨이가 캐시 파일 '경로'를 도구 인자로 넘기는 관습에 대응(R2). 추출 로직은
test_face.py가 검증하므로 여기선 입력 해석만 본다."""
import base64
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import server  # noqa: E402  (FastMCP tool 엔트리)
from face_core import read_image_input  # noqa: E402

HERE = os.path.dirname(__file__)
FACE = os.path.join(HERE, "assets", "face_sample.png")


def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def test_read_image_input_path_returns_file_bytes():
    with open(FACE, "rb") as f:
        raw = f.read()
    assert read_image_input(FACE) == raw  # 경로 → 파일 바이트 그대로


def test_path_input_is_person():
    r = server.analyze_face(FACE)  # 로컬 경로 입력(게이트웨이 핸드오프 형태)
    assert r["is_person"] is True
    assert "samjeong" in r["features"]


def test_base64_input_is_person_regression():
    r = server.analyze_face(_b64(FACE))
    assert r["is_person"] is True


def test_data_uri_input_is_person():
    r = server.analyze_face("data:image/png;base64," + _b64(FACE))
    assert r["is_person"] is True


def test_path_and_base64_same_output():
    a = server.analyze_face(FACE)
    b = server.analyze_face(_b64(FACE))
    assert a == b  # 경로·base64 동일 입력 → 동일 출력(결정론)


def test_nonexistent_path_no_crash():
    r = server.analyze_face("/no/such/dir/ghost.jpg")
    assert r["is_person"] is False
    assert "reason" in r


def test_non_image_file_no_crash():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as t:
        t.write(b"this is plain text, not an image")
        p = t.name
    try:
        r = server.analyze_face(p)  # 실재 파일이나 이미지 아님
        assert r["is_person"] is False
        assert "reason" in r
    finally:
        os.unlink(p)


def test_empty_input():
    r = server.analyze_face("   ")
    assert r["is_person"] is False
