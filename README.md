# mcp-gwansang

관상(觀相) **추출** MCP — 사진에서 결정론적 얼굴 피처를 뽑아 구조 JSON으로 돌려준다.
SATI 패턴의 **손발(추출)** 만 담당한다. **관상 해석·점수·운세는 만들지 않는다(AI 몫).**

## 원칙
- **외부 API 0.** 전부 로컬(MediaPipe Face Mesh, Apache-2.0). 사진을 어떤 클라우드로도 보내지 않는다.
- **순수 추출.** 三停/五官 등 관상 *부위명*을 키로 쓰지만 의미 부여는 안 한다 — 측정치(비율·각도)만 반환.
- **단일 tool · 단일 호출.** 한 번에 완전한 피처를 돌려줘 소형 모델의 다중 tool 호출을 피한다.

## tool
### `analyze_face(image, include_overlay=False)`
- `image`: 얼굴 사진. data URI(`data:image/jpeg;base64,...`) 또는 순수 base64.
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
    "samjeong": { "upper": 0.33, "middle": 0.34, "lower": 0.33 },
    "ogwan": {
      "eye_length_ratio": 0.22, "interocular_ratio": 0.21,
      "nose_width_ratio": 0.27, "nose_length_ratio": 0.26,
      "mouth_width_ratio": 0.36, "brow_eye_span_ratio": 0.07
    },
    "face_shape_ratio": 1.38,
    "symmetry": 0.96,
    "pose": { "yaw_balance": 0.97, "roll_deg": -1.2 },
    "geometry_px": { "face_width": 612.0, "face_height": 845.0, "image_w": 1024, "image_h": 1024 }
  },
  "confidence": "high",
  "notes": []
}
```
`confidence: "low"`면 `notes`에 사유(측면·흐림·작은 얼굴·기울어짐). v0은 고신뢰 코어만 — 十二宫 좌표매핑·气색·귀(耳)는 제외(후속).

## 실행
```bash
~/.local/bin/python3.11 -m venv venv
./venv/bin/python -m pip install -r requirements.txt
./venv/bin/python server.py   # stdio MCP
```
MCP 등록(헤르메스): command=`<repo>/venv/bin/python`, args=`["<repo>/server.py"]`, env `{}`.

## 프라이버시
사진은 메모리에서 MediaPipe로만 처리하고 디스크/네트워크로 내보내지 않는다. 저장 없음.

MIT · molpass
