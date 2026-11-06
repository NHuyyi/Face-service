"""Ví dụ gọi service: enroll người 1 -> match người 2. Dùng: python scripts/client_example.py a.jpg b.jpg"""
import sys

import requests  # pip install requests

BASE = "http://localhost:8000"
HEAD = {}  # {"X-API-Key": "..."} nếu bật API_KEY

a, b = sys.argv[1:3]
r = requests.post(f"{BASE}/v1/faces", files={"file": open(a, "rb")}, data={"label": "nguoi_1"}, headers=HEAD)
r.raise_for_status()
face_id = r.json()["id"]
print("Đã lưu vector 512-d, id =", face_id)

m = requests.post(f"{BASE}/v1/faces/{face_id}/match", files={"file": open(b, "rb")}, headers=HEAD).json()
print(m["result"], "- similarity =", round(m["similarity"], 4))
