# -*- coding: utf-8 -*-
"""mcp-gwansang 추출 코어 테스트 — 사람/비사람 게이트·범위·결정론. 해석은 검증 대상 아님."""
import base64
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from face_core import analyze, decode_base64_image  # noqa: E402

HERE = os.path.dirname(__file__)
FACE = os.path.join(HERE, "assets", "face_sample.png")


def _png_bytes(arr):
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    return buf.tobytes()


# ── 비사람 게이트 ──
def test_solid_image_not_person():
    r = analyze(_png_bytes(np.full((400, 400, 3), 127, np.uint8)))
    assert r["is_person"] is False
    assert "reason" in r


def test_noise_image_not_person():
    rng = np.random.RandomState(0)
    r = analyze(_png_bytes(rng.randint(0, 255, (400, 400, 3), dtype=np.uint8)))
    assert r["is_person"] is False


def test_garbage_bytes_not_person():
    r = analyze(b"this is not an image")
    assert r["is_person"] is False
    assert "디코드" in r["reason"]


# ── base64 헬퍼 ──
def test_base64_helper_data_uri_and_raw():
    raw = b"\x89PNG\r\n payload"
    enc = base64.b64encode(raw).decode()
    assert decode_base64_image("data:image/png;base64," + enc) == raw
    assert decode_base64_image(enc) == raw


# ── 사람 피처(실제 얼굴 fixture) ──
@pytest.mark.skipif(not os.path.exists(FACE), reason="face fixture 없음")
def test_face_is_person_and_feature_shape():
    with open(FACE, "rb") as f:
        r = analyze(f.read())
    assert r["is_person"] is True
    f = r["features"]
    # 三停 비율 합 = 1, 각 (0,1)
    sj = f["samjeong"]
    assert abs(sj["upper"] + sj["middle"] + sj["lower"] - 1.0) < 0.01
    for v in sj.values():
        assert 0.0 < v < 1.0
    # 五官 비율 — 상식 범위
    og = f["ogwan"]
    assert 0.05 < og["eye_length_ratio"] < 0.40
    assert 0.10 < og["interocular_ratio"] < 0.50
    assert 0.10 < og["nose_width_ratio"] < 0.60
    assert 0.10 < og["mouth_width_ratio"] < 0.70
    # 대칭/자세
    assert 0.0 <= f["symmetry"] <= 1.0
    assert 0.0 < f["pose"]["yaw_balance"] <= 1.0
    assert r["confidence"] in ("high", "low")
    assert isinstance(r["notes"], list)


@pytest.mark.skipif(not os.path.exists(FACE), reason="face fixture 없음")
def test_determinism_same_image_same_output():
    with open(FACE, "rb") as f:
        data = f.read()
    assert analyze(data) == analyze(data)


@pytest.mark.skipif(not os.path.exists(FACE), reason="face fixture 없음")
def test_overlay_optional_png():
    with open(FACE, "rb") as f:
        r = analyze(f.read(), overlay=True)
    assert r["is_person"] is True
    assert "overlay_png_base64" in r
    # PNG 시그니처 확인
    head = base64.b64decode(r["overlay_png_base64"])[:8]
    assert head[:4] == b"\x89PNG"
