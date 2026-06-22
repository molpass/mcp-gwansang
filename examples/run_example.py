# -*- coding: utf-8 -*-
"""예시 실행 — 얼굴 fixture/비사람 입력을 analyze()에 넣어 구조 JSON을 출력·저장.
사용: <venv>/python examples/run_example.py
"""
import json
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from face_core import analyze  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FACE = os.path.join(HERE, "..", "tests", "assets", "face_sample.png")


def main():
    with open(FACE, "rb") as f:
        face = analyze(f.read())
    ok, buf = cv2.imencode(".png", np.full((300, 300, 3), 127, np.uint8))
    nonperson = analyze(buf.tobytes())

    out = {"person_example": face, "non_person_example": nonperson}
    dst = os.path.join(HERE, "sample_output.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("\nsaved:", dst)


if __name__ == "__main__":
    main()
