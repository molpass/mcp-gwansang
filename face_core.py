# -*- coding: utf-8 -*-
"""mcp-gwansang — 관상 추출 코어 (결정론 얼굴 피처).

손발 전용: 사진 → 사람 판별 → MediaPipe Face Mesh 468점 → 정규화 구조 피처(JSON).
해석·점수·운세 문구 절대 없음(AI 몫). 외부 API 0 — 전부 로컬, 사진 외부전송 0.
"""
from __future__ import annotations

import base64
import math
import os
from typing import Optional

import cv2
import numpy as np

# MediaPipe Tasks(FaceLandmarker) — 번들 .task 모델(로컬, 오프라인). 외부 다운로드 0.
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
_MODEL_PATH = os.path.join(_ASSETS, "face_landmarker.task")
_SEG_MODEL_PATH = os.path.join(_ASSETS, "selfie_multiclass_256x256.tflite")  # 발제선 보정용(로컬·오프라인)
_landmarker = None  # 모델 로드 1회 재사용
_segmenter = None


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH),
            num_faces=1,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        _landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
    return _landmarker


def _get_segmenter():
    global _segmenter
    if _segmenter is None:
        opts = mp_vision.ImageSegmenterOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=_SEG_MODEL_PATH),
            running_mode=mp_vision.RunningMode.IMAGE,
            output_category_mask=True,
        )
        _segmenter = mp_vision.ImageSegmenter.create_from_options(opts)
    return _segmenter

# ── 사용 랜드마크 인덱스(468 mesh). 값에 직접 쓰는 점만 명시 — 검증/오버레이 대상. ──
# 三停 경계
IDX_FOREHEAD_TOP = 10   # 발제선 근사(이마 최상단)
IDX_CHIN = 152          # 턱끝
IDX_NOSE_TIP = 1        # 코끝(준두)
IDX_NASION = 168        # 코뿌리(미간 아래) — 코길이/중심선
IDX_BROW_L = 105        # 왼눈썹 중앙
IDX_BROW_R = 334        # 오른눈썹 중앙
# 五官
IDX_EYE_L_OUT, IDX_EYE_L_IN = 33, 133    # 왼눈 바깥/안쪽
IDX_EYE_R_IN, IDX_EYE_R_OUT = 362, 263   # 오른눈 안쪽/바깥
IDX_MOUTH_L, IDX_MOUTH_R = 61, 291       # 입꼬리
IDX_ALA_L, IDX_ALA_R = 48, 278           # 콧방울(코너비)
# 얼굴폭(광대/볼 라인)
IDX_CHEEK_L, IDX_CHEEK_R = 234, 454

# 대칭 평가용 좌우 대칭쌍
SYMMETRY_PAIRS = [
    (IDX_EYE_L_OUT, IDX_EYE_R_OUT),
    (IDX_MOUTH_L, IDX_MOUTH_R),
    (IDX_ALA_L, IDX_ALA_R),
    (IDX_CHEEK_L, IDX_CHEEK_R),
]

# 품질 게이트 임계값
MIN_FACE_PX = 90.0       # 얼굴 세로 픽셀 하한(미만이면 저신뢰)
BLUR_MIN = 60.0          # 라플라시안 분산 하한(미만이면 흐림)
YAW_BALANCE_MIN = 0.80   # 좌우 반폭 비(미만이면 측면 의심)
ROLL_MAX_DEG = 12.0      # 기울기 상한

# 발제선(髮際線) 보정 — selfie_multiclass 세그멘테이션 클래스 인덱스
_HAIR_CLASS = 1          # 머리카락
_SKIN_CLASS = 3          # 얼굴 피부


def _estimate_hairline(mp_image, w: int, h: int, midx: int, brow_y: float):
    """정중선에서 머리카락↔이마피부 경계(발제선)를 찾는다.

    랜드마크 10번(메시 이마 최상단)은 실제 발제선보다 보통 아래라 上停이 과소측정된다
    (전형 정면 검증 7/7). 세그멘테이션으로 실제 발제선을 잡아 보정한다.

    반환: 성공 → (hairline_y, "segment"). 실패(세그 불가·대머리·이마 가림·앞머리) → (None, "fallback").
    실패는 모델 부재·런타임 오류까지 흡수해 폴백으로 회귀(클론이 모델 미보유여도 동작).
    """
    try:
        mask = _get_segmenter().segment(mp_image).category_mask.numpy_view().reshape(h, w)
    except Exception:  # noqa: BLE001
        return None, "fallback"
    lo = max(0, midx - 3)
    hi = min(w, midx + 4)
    band = mask[:, lo:hi]
    seen_skin = False
    y = int(brow_y) - 1
    while y > 0:
        row = band[y]
        vals, counts = np.unique(row, return_counts=True)
        c = int(vals[int(np.argmax(counts))])  # 정중선 밴드의 최빈 클래스
        if c == _SKIN_CLASS:
            seen_skin = True
        elif c == _HAIR_CLASS and seen_skin:
            return float(y + 1), "segment"  # 이마피부 위로 머리카락이 시작되는 경계 = 발제선
        y -= 1
    return None, "fallback"


def _decode_image(image_bytes: bytes) -> Optional[np.ndarray]:
    """바이트 → BGR ndarray. 디코드 실패 시 None."""
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def decode_base64_image(data: str) -> bytes:
    """data URI(`data:image/...;base64,XXXX`) 또는 순수 base64 → bytes."""
    s = data.strip()
    if s.startswith("data:"):
        comma = s.find(",")
        if comma != -1:
            s = s[comma + 1 :]
    return base64.b64decode(s)


def read_image_input(image: str) -> bytes:
    """image 인자 → 바이트. 헤르메스 게이트웨이는 '바이트는 캐시 저장, 도구엔 경로' 관습이라
    로컬 파일 경로도 받는다. 순서: (1) 존재하는 로컬 파일 경로 → 파일 바이트
    (2) data-URI / 순수 base64 → 디코드. base64는 길고 슬래시·null이 없어 경로 오인 위험 없음
    (긴 문자열은 isfile False 또는 path-too-long으로 자연히 (2)로 폴백)."""
    s = image.strip()
    # 1) 로컬 파일 경로 — 짧고(경로 길이 한계 내) 실재하는 일반 파일일 때만.
    if len(s) < 4096 and os.path.isfile(s):
        with open(s, "rb") as f:
            return f.read()
    # 2) data-URI / base64
    return decode_base64_image(s)


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _round(v: float, n: int = 4) -> float:
    return round(float(v), n)


def analyze(image_bytes: bytes, overlay: bool = False) -> dict:
    """사진 바이트 → 구조 피처 JSON(dict). 해석 없음.

    반환:
      사람 없음 → {is_person: False, reason}
      사람 있음 → {is_person: True, features{...}, confidence, notes, [overlay_png_base64]}
    """
    img = _decode_image(image_bytes)
    if img is None:
        return {"is_person": False, "reason": "이미지 디코드 실패(지원 형식 아님)"}

    h, w = img.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = _get_landmarker().detect(mp_image)

    if not res.face_landmarks:
        return {"is_person": False, "reason": "얼굴이 검출되지 않음(사람 아님 또는 얼굴 미포함)"}

    lm = res.face_landmarks[0]

    def P(idx: int):
        return (lm[idx].x * w, lm[idx].y * h)

    # ── 기준 좌표 ──
    top = P(IDX_FOREHEAD_TOP)
    chin = P(IDX_CHIN)
    nose_tip = P(IDX_NOSE_TIP)
    nasion = P(IDX_NASION)
    brow = ((P(IDX_BROW_L)[0] + P(IDX_BROW_R)[0]) / 2, (P(IDX_BROW_L)[1] + P(IDX_BROW_R)[1]) / 2)
    cheek_l, cheek_r = P(IDX_CHEEK_L), P(IDX_CHEEK_R)

    face_width = abs(cheek_r[0] - cheek_l[0])
    face_height_raw = abs(chin[1] - top[1])  # 메시-top 기준(보정 전)
    if face_width < 1 or face_height_raw < 1:
        return {"is_person": False, "reason": "얼굴 기하 산출 불가(퇴화된 좌표)"}

    # ── 발제선 보정 ──
    # 발제선 추정 성공 시 上停·face_height의 상단 기준을 메시-top → 발제선으로 교체.
    # 높이 정규화 피처(face_shape_ratio·nose_length_ratio·brow_eye_span_ratio)가 함께 정상화된다.
    midx = int(round((top[0] + brow[0]) / 2))
    hairline_y, hairline_method = _estimate_hairline(mp_image, w, h, midx, brow[1])
    hairline_estimated = hairline_method == "segment"
    top_y = hairline_y if hairline_estimated else top[1]
    face_height = abs(chin[1] - top_y)

    # ── 三停(세로 3등분 비율) — 발제선 기준 ──
    upper = brow[1] - top_y
    mid = nose_tip[1] - brow[1]
    lower = chin[1] - nose_tip[1]
    total = upper + mid + lower
    samjeong = {
        "upper": _round(upper / total),
        "middle": _round(mid / total),
        "lower": _round(lower / total),
    }
    # 上停 raw(옛 메시-top 기준) — 핑거프린트·진위검증·해석층 비교용 보존
    upper_raw = brow[1] - top[1]
    total_raw = upper_raw + mid + lower
    samjeong_meta = {
        "upper_raw": _round(upper_raw / total_raw),
        "hairline_estimated": hairline_estimated,
        "hairline_method": hairline_method,
    }

    # ── 五官 비율(얼굴폭/높이 대비) ──
    eye_l_len = _dist(P(IDX_EYE_L_OUT), P(IDX_EYE_L_IN))
    eye_r_len = _dist(P(IDX_EYE_R_IN), P(IDX_EYE_R_OUT))
    eye_len = (eye_l_len + eye_r_len) / 2
    interocular = _dist(P(IDX_EYE_L_IN), P(IDX_EYE_R_IN))
    nose_width = _dist(P(IDX_ALA_L), P(IDX_ALA_R))
    mouth_width = _dist(P(IDX_MOUTH_L), P(IDX_MOUTH_R))
    nose_length = _dist(nasion, nose_tip)
    brow_len = (_dist(P(IDX_BROW_L), P(IDX_EYE_L_OUT)) + _dist(P(IDX_BROW_R), P(IDX_EYE_R_OUT))) / 2

    ogwan = {
        "eye_length_ratio": _round(eye_len / face_width),       # 三庭五眼: ~0.2 근사
        "interocular_ratio": _round(interocular / face_width),  # 양안 간격
        "nose_width_ratio": _round(nose_width / face_width),
        "nose_length_ratio": _round(nose_length / face_height),
        "mouth_width_ratio": _round(mouth_width / face_width),
        "brow_eye_span_ratio": _round(brow_len / face_height),  # 눈썹~눈 거리(세로)
    }

    # ── 얼굴형 / 대칭 ──
    face_shape_ratio = _round(face_height / face_width)  # 세로/가로

    midline_x = (top[0] + chin[0] + nasion[0] + nose_tip[0]) / 4
    devs = []
    for li, ri in SYMMETRY_PAIRS:
        lp, rp = P(li), P(ri)
        left_off = abs(midline_x - lp[0])
        right_off = abs(rp[0] - midline_x)
        devs.append(abs(left_off - right_off) / face_width)
    symmetry = _round(max(0.0, 1.0 - (sum(devs) / len(devs))))

    # ── 자세(정면도) ──
    left_half = abs(nose_tip[0] - cheek_l[0])
    right_half = abs(cheek_r[0] - nose_tip[0])
    yaw_balance = _round(min(left_half, right_half) / max(left_half, right_half))  # 1.0=정면
    roll_deg = _round(
        math.degrees(math.atan2(P(IDX_EYE_R_OUT)[1] - P(IDX_EYE_L_OUT)[1],
                                P(IDX_EYE_R_OUT)[0] - P(IDX_EYE_L_OUT)[0])), 2)

    # ── 선명도(흐림) — 얼굴 ROI 라플라시안 분산 ──
    x0 = max(0, int(min(cheek_l[0], cheek_r[0])))
    x1 = min(w, int(max(cheek_l[0], cheek_r[0])))
    y0 = max(0, int(top[1]))
    y1 = min(h, int(chin[1]))
    roi = img[y0:y1, x0:x1]
    blur = float(cv2.Laplacian(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()) if roi.size else 0.0

    # ── 신뢰도 게이트(저신뢰 사유 수집) ──
    # 이미지 품질 게이트는 confidence(전역)를 좌우한다. 발제선 폴백은 上停 축만 저신뢰이므로
    # samjeong_meta로 알리고 안내 note만 덧붙인다 — 정상 사진이면 五官·대칭 해석은 유효하게 둔다.
    quality_notes = []
    if face_height_raw < MIN_FACE_PX:
        quality_notes.append(f"얼굴이 작음({int(face_height_raw)}px<{int(MIN_FACE_PX)}) — 피처 신뢰 낮음")
    if blur < BLUR_MIN:
        quality_notes.append(f"흐림(라플라시안 분산 {int(blur)}<{int(BLUR_MIN)})")
    if yaw_balance < YAW_BALANCE_MIN:
        quality_notes.append(f"측면 의심(좌우 반폭비 {yaw_balance}<{YAW_BALANCE_MIN})")
    if abs(roll_deg) > ROLL_MAX_DEG:
        quality_notes.append(f"기울어짐(roll {roll_deg}°)")
    confidence = "low" if quality_notes else "high"
    notes = list(quality_notes)
    if not hairline_estimated:
        notes.append("발제선 추정 실패 — 상정(上停) 보정 불가(이마 상단 기준), 상정축 저신뢰")

    result = {
        "is_person": True,
        "features": {
            "samjeong": samjeong,          # 三停 비율(상/중/하) — 上停은 발제선 보정 반영
            "samjeong_meta": samjeong_meta,  # upper_raw(보정 전)·발제선 추정 여부·방식
            "ogwan": ogwan,                # 五官 비율
            "face_shape_ratio": face_shape_ratio,
            "symmetry": symmetry,          # 0~1 (1=완전대칭)
            "pose": {"yaw_balance": yaw_balance, "roll_deg": roll_deg},
            "geometry_px": {
                "face_width": _round(face_width, 1),
                "face_height": _round(face_height, 1),       # 발제선 보정 기준
                "face_height_raw": _round(face_height_raw, 1),  # 메시-top 기준(보정 전)
                "image_w": w, "image_h": h,
            },
        },
        "confidence": confidence,
        "notes": notes,
    }

    if overlay:
        result["overlay_png_base64"] = _make_overlay(img, lm, w, h)
    return result


def _make_overlay(img: np.ndarray, lm, w: int, h: int) -> str:
    """사용 랜드마크를 점으로 찍은 검증용 PNG(base64). CJK 텍스트 없음 — 폰트 불필요."""
    canvas = img.copy()
    used = [
        IDX_FOREHEAD_TOP, IDX_CHIN, IDX_NOSE_TIP, IDX_NASION, IDX_BROW_L, IDX_BROW_R,
        IDX_EYE_L_OUT, IDX_EYE_L_IN, IDX_EYE_R_IN, IDX_EYE_R_OUT,
        IDX_MOUTH_L, IDX_MOUTH_R, IDX_ALA_L, IDX_ALA_R, IDX_CHEEK_L, IDX_CHEEK_R,
    ]
    for idx in used:
        cx, cy = int(lm[idx].x * w), int(lm[idx].y * h)
        cv2.circle(canvas, (cx, cy), 3, (0, 255, 0), -1)
    ok, png = cv2.imencode(".png", canvas)
    return base64.b64encode(png.tobytes()).decode("ascii") if ok else ""
