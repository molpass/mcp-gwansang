# mcp-gwansang

관상(觀相) **추출** MCP — 사진에서 결정론적 얼굴 피처를 뽑아 구조 JSON으로 돌려준다.
SATI 패턴의 **손발(추출)** 만 담당한다. **관상 해석·점수·운세는 만들지 않는다(AI 몫).**

## 원칙
- **외부 API 0.** 전부 로컬(MediaPipe FaceLandmarker + 발제선 보정용 selfie_multiclass 세그멘테이션, Apache-2.0). 사진을 어떤 클라우드로도 보내지 않는다.
- **순수 추출.** 三停/五官 등 관상 *부위명*을 키로 쓰지만 의미 부여는 안 한다 — 측정치(비율·각도)만 반환.
- **단일 tool · 단일 호출.** 한 번에 완전한 피처를 돌려줘 소형 모델의 다중 tool 호출을 피한다.

## tool
### `analyze_face(image, include_overlay=False)`
- `image`: 얼굴 사진. 로컬 파일 경로 / data URI(`data:image/jpeg;base64,...`) / 순수 base64.
- `include_overlay`: True면 사용 랜드마크를 찍은 검증용 PNG(base64) 동봉(텍스트 없음 — CJK 폰트 불필요).

#### 반환(JSON)
사람 없음:
```json
{ "is_person": false, "reason": "얼굴이 검출되지 않음(사람 아님 또는 얼굴 미포함)" }
```
사람 있음:
```json
{
  "is_person": true,
  "features": {
    "samjeong": { "upper": 0.34, "middle": 0.30, "lower": 0.36 },
    "samjeong_meta": { "upper_raw": 0.17, "hairline_estimated": true, "hairline_method": "segment" },
    "ogwan": {
      "eye_length_ratio": 0.18, "interocular_ratio": 0.25,
      "nose_width_ratio": 0.27, "nose_length_ratio": 0.19,
      "mouth_width_ratio": 0.39, "brow_eye_span_ratio": 0.11
    },
    "face_shape_ratio": 1.46,
    "symmetry": 0.94,
    "pose": { "yaw_balance": 0.87, "roll_deg": -9.5 },
    "geometry_px": { "face_width": 79.4, "face_height": 115.6, "face_height_raw": 92.3, "image_w": 512, "image_h": 640 }
  },
  "confidence": "high",
  "notes": []
}
```
**발제선(髮際線) 보정**: 랜드마크 메시 최상단(idx 10)은 실제 발제선보다 보통 아래라 상정(上停)이 과소측정된다(전형 정면 검증 7/7). 세그멘테이션으로 정중선의 머리카락↔이마 경계를 잡아 상정·`face_height`를 재계산한다(높이 정규화 피처도 함께 정상화). `samjeong_meta.hairline_method`: `segment`(보정 성공) | `fallback`(대머리·이마 가림·세그 실패 → 메시-top 회귀, `hairline_estimated:false`, `notes`에 안내·상정 축만 저신뢰). `upper_raw`는 보정 전 값(진위검증·해석층 비교용). `face_height_raw`는 보정 전 높이.

`confidence: "low"`면 `notes`에 사유(측면·흐림·작은 얼굴·기울어짐). v0은 고신뢰 코어만 — 십이궁(十二宮) 좌표매핑·기색(氣色)·귀(耳)는 제외(후속).

## 실행
```bash
~/.local/bin/python3.11 -m venv venv
./venv/bin/python -m pip install -r requirements.txt
./venv/bin/python server.py   # stdio MCP
```
MCP 등록(헤르메스): command=`<repo>/venv/bin/python`, args=`["<repo>/server.py"]`, env `{}`.

## 헤르메스 SKILL 설치 (해석층)
이 레포는 **손발(추출 MCP)** 과 **지식(해석 SKILL)** 을 함께 버전 관리한다. 클론 한 번으로 둘 다 따라온다.
`skill/SKILL.md`는 측정치를 재현 가능·전거 있는 관상 해석으로 옮기는 규율(측정→해석 매핑·전거 인용·도구 실패 시 추측 차단)을 담은 **작성된 정적 산출물**이다. 헤르메스/모델이 런타임에 스스로 해석을 만들지 않는다.

설치(회사·아슬 동일 — 기기 비의존):
1. 레포 클론 + venv + requirements (위 [실행](#실행)).
2. MCP 등록 (위 — `command=venv/python`, `args=[server.py]`).
3. `skill/SKILL.md`를 헤르메스 스킬 디렉터리에 복사:
   ```bash
   mkdir -p ~/.hermes/skills/gwansang
   cp skill/SKILL.md ~/.hermes/skills/gwansang/SKILL.md
   # 레포 추적을 유지하려면 심볼릭 링크:
   # ln -sf "$(pwd)/skill/SKILL.md" ~/.hermes/skills/gwansang/SKILL.md
   ```
4. 게이트웨이 재시작으로 스킬 로드.

## 프라이버시
사진은 메모리에서 MediaPipe로만 처리하고 디스크/네트워크로 내보내지 않는다. 저장 없음.

MIT · molpass
