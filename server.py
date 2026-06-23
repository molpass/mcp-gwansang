#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""mcp-gwansang — 관상 추출 MCP 서버 (stdio).

analyze_face: 사진(base64) → 사람 판별 + 결정론 얼굴 피처(三停·五官 비율·대칭·자세) JSON.
순수 추출(손발) — 관상 해석·점수·운세는 절대 생성하지 않는다(AI 몫).
외부 API 0 · 사진 외부전송 0 · 단일 tool 단일 호출(한 번에 완전 반환).
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from face_core import analyze, read_image_input

mcp = FastMCP("gwansang")


@mcp.tool()
def analyze_face(image: str, include_overlay: bool = False) -> dict:
    """사진에서 결정론 얼굴 피처를 추출한다(해석 없음 — 수치만).

    Args:
        image: 얼굴 사진. 로컬 파일 경로 / data URI(`data:image/jpeg;base64,...`) / 순수 base64.
        include_overlay: True면 사용 랜드마크를 찍은 검증용 PNG(base64)를 함께 반환.

    Returns:
        사람 없음 → {"is_person": false, "reason": ...}
        사람 있음 → {"is_person": true, "features": {samjeong(三停 비율),
        ogwan(五官 비율), face_shape_ratio, symmetry, pose, geometry_px},
        "confidence": "high"|"low", "notes": [...]}.
        ※ 이 값들은 해석이 아니라 측정치다. 관상 의미 부여는 호출한 AI가 한다.
    """
    if not isinstance(image, str) or not image.strip():
        return {"is_person": False, "reason": "image(경로 또는 base64)가 비어 있음"}
    try:
        raw = read_image_input(image)
    except Exception as e:  # noqa: BLE001 — 입력 오류를 명확 메시지로
        return {"is_person": False, "reason": f"이미지 입력 처리 실패: {e}"}
    return analyze(raw, overlay=include_overlay)


if __name__ == "__main__":
    mcp.run()
