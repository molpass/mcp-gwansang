# mcp-gwansang

관상(觀相) **추출** MCP — 사진에서 결정론적 얼굴 피처를 뽑아 구조 JSON으로 돌려준다.
MCP-추출, SKILL-지식, AI-해석 합체 패턴으로 MCP는**손발(추출)** 만 담당한다. **관상 해석·점수·운세는 만들지 않는다(그것은 AI 몫).**

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

## 왜 SKILL을 함께 주는가 (손발 + 지식)
이 레포는 **손발(추출 MCP)** 과 **지식(해석 SKILL)** 을 한 벌로 버전 관리한다. 클론 한 번에 둘 다 따라온다.

MCP만 쓰면 모델이 측정 수치를 **즉흥 해석**한다 — 재현성도 전거도 없고, 도구가 실패하면 사진을 눈으로 보고 지어내는(환각) 모드로 샌다. 그래서 해석을 `skill/`의 **정적 SKILL**로 고정한다:
- **① 전거 인용** — 모든 해석을 측정 키·값에 묶는다(못 묶으면 말하지 않음).
- **② 추측 차단** — 도구 실패·결과 잘림이면 해석 0줄, 재촬영 안내(비전 추측 금지).
- **③ 재현성** — 같은 사진 → 같은 측정 → 같은 해석 틀.

### 2변형 — 용도에 맞게 하나를 깐다
| 변형 | 용도 | 표현 |
|---|---|---|
| `skill/SKILL.casual.md` | 게스트·운세 경험 | 쉬운 일상어만 — 숫자·전문용어·한자 표면 0 |
| `skill/SKILL.expert.md` | 학습·전문가·관리자 | 마의상법(麻衣相法) 전거·수치 노출 |

안티-환각 규율(전거 인용·추측 차단·미측정 부위 언급 금지·면책)은 **두 변형 공통**, 차이는 표현층뿐. 어느 환경에 무엇을 깔지는 배포 선택이다.

### 설치 (회사·홈 동일 — 기기 비의존)
1. 레포 클론 + venv + requirements (위 [실행](#실행)).
2. MCP 등록 (위 — `command=venv/python`, `args=[server.py]`).
3. 용도에 맞는 변형을 헤르메스 스킬 디렉터리에 `SKILL.md`로 복사:
   ```bash
   mkdir -p <skills>/gwansang
   cp skill/SKILL.casual.md  <skills>/gwansang/SKILL.md   # 게스트용
   # 또는
   cp skill/SKILL.expert.md  <skills>/gwansang/SKILL.md   # 전문가용
   ```
   (`<skills>`는 해당 프로필의 스킬 디렉터리.)
4. 게이트웨이 재시작으로 스킬 로드.

## 프라이버시·면책
사진은 메모리에서 MediaPipe로만 처리하고 디스크/네트워크로 내보내지 않는다. **저장·전송 없음.** 관상은 얼굴 비율 측정에 기반한 **참고용 풀이**이며 운명론이 아니다(과학적 사실 아님).

## About / 제작
**Hermes Agent용 MCP** — molpass의 바이브 코딩(vibe coding) 프로젝트.

- 아이디어·방향: **molpass (이정훈)** · https://zeolinex.com
- 기획: **Claude (Chat)**
- 개발: **Claude Code**

같은 모음:
- [mcp-saju](https://github.com/molpass/mcp-saju) · [mcp-qr](https://github.com/molpass/mcp-qr) · [mcp-biorhythm](https://github.com/molpass/mcp-biorhythm) · [mcp-astrology](https://github.com/molpass/mcp-astrology) · [mcp-ziwei](https://github.com/molpass/mcp-ziwei) · [mcp-numerology](https://github.com/molpass/mcp-numerology) · [mcp-liuren](https://github.com/molpass/mcp-liuren) · [mcp-qimen](https://github.com/molpass/mcp-qimen) · [mcp-taiyi](https://github.com/molpass/mcp-taiyi) · [mcp-weather](https://github.com/molpass/mcp-weather) · [mcp-newsfeed](https://github.com/molpass/mcp-newsfeed) · [mcp-bible](https://github.com/molpass/mcp-bible)
- **mcp-gwansang** (이 repo)

## License
MIT (코드) — molpass. 모델·데이터 출처:
- MediaPipe **FaceLandmarker** · **Selfie Multiclass Segmentation** (Google) — Apache-2.0. 모델은 로컬 동봉, 오프라인 추론.
