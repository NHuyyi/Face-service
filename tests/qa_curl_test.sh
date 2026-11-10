#!/usr/bin/env bash
# =============================================================================
# Face Recognition Service – QA Test Plan (curl)
# Bao phủ:
#   - Luồng 2 bước: enroll → match (cùng người / khác người)
#   - /v1/compare  : so trực tiếp 2 ảnh
#   - /v1/identify : 1:N trong DB
#   - /v1/embeddings: lấy vector không lưu DB
#   - Các lỗi: 400 (ảnh hỏng), 400 (rỗng), 413 (quá lớn), 422 (không thấy mặt), 404 (id không tồn tại)
#   - Xác thực API key (401)
#   - GET /health, GET /v1/faces, DELETE /v1/faces/{id}
#
# Yêu cầu:
#   - curl, jq, python3 (tạo ảnh giả), bc
#   - Service chạy tại BASE_URL
#   - Có ít nhất 2 ảnh người thật: TEST_IMG_A và TEST_IMG_B (cùng người)
#                                   TEST_IMG_C (người khác)
# =============================================================================

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
API_KEY="${API_KEY:-}"          # Để trống nếu không dùng API key
TEST_IMG_A="${TEST_IMG_A:-tests/assets/person_a_1.jpg}"   # Ảnh người A (ảnh 1)
TEST_IMG_B="${TEST_IMG_B:-tests/assets/person_a_2.jpg}"   # Ảnh người A (ảnh 2)
TEST_IMG_C="${TEST_IMG_C:-tests/assets/person_b_1.jpg}"   # Ảnh người B (khác người)
MATCH_THRESHOLD="${MATCH_THRESHOLD:-0.40}"

PASS=0
FAIL=0
SKIP=0

# ---------- helpers ----------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

log_pass() { echo -e "${GREEN}[PASS]${NC} $1"; ((PASS++)); }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; ((FAIL++)); }
log_skip() { echo -e "${YELLOW}[SKIP]${NC} $1"; ((SKIP++)); }
log_info() { echo -e "       $1"; }

auth_header() {
  if [ -n "$API_KEY" ]; then echo "-H 'X-API-Key: $API_KEY'"; else echo ""; fi
}

curl_post() {
  eval curl -s -w '\n%{http_code}' $(auth_header) "$@"
}

check_status() {
  local name="$1" expected="$2" actual="$3"
  if [ "$actual" -eq "$expected" ]; then
    log_pass "$name → HTTP $actual"
  else
    log_fail "$name → kỳ vọng HTTP $expected, nhận HTTP $actual"
  fi
}

check_json_field() {
  local name="$1" json="$2" field="$3" expected="$4"
  local actual
  actual=$(echo "$json" | jq -r "$field" 2>/dev/null || echo "ERROR_JQ")
  if [ "$actual" = "$expected" ]; then
    log_pass "$name: $field = $actual"
  else
    log_fail "$name: $field kỳ vọng '$expected', nhận '$actual'"
  fi
}

# Tạo ảnh giả bằng Python khi không có ảnh thật
make_dummy_img() {
  local path="$1" value="$2"
  python3 - <<EOF
import cv2, numpy as np, os
os.makedirs(os.path.dirname("$path") or ".", exist_ok=True)
cv2.imwrite("$path", np.full((200, 200, 3), $value, np.uint8))
print("Created dummy: $path")
EOF
}

# ---------- setup -------------------------------------------------------------
echo "=============================================="
echo "  Face Service QA – curl Test Plan"
echo "  BASE_URL = $BASE_URL"
echo "  $(date)"
echo "=============================================="

# Tạo ảnh dummy nếu chưa có (thay thế bằng ảnh thật để kiểm tra đầy đủ)
mkdir -p tests/assets
for f in "$TEST_IMG_A" "$TEST_IMG_B" "$TEST_IMG_C"; do
  if [ ! -f "$f" ]; then
    log_info "Ảnh $f chưa tồn tại → tạo ảnh dummy (xám)"
    make_dummy_img "$f" 128
    log_info "⚠️  Thay bằng ảnh người thật để kiểm tra match chính xác"
  fi
done

# Tạo ảnh trắng (gây lỗi 422 "không thấy mặt")
BLANK_IMG="tests/assets/blank_white.jpg"
[ ! -f "$BLANK_IMG" ] && python3 -c "
import cv2, numpy as np; cv2.imwrite('$BLANK_IMG', np.full((64,64,3),255,np.uint8))
"

# Tạo file text giả (gây lỗi 400 "định dạng không hỗ trợ")
FAKE_TXT="tests/assets/fake.txt"
echo "not an image" > "$FAKE_TXT"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  TC-01  GET /health"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
RESP=$(curl -s -w '\n%{http_code}' "$BASE_URL/health")
STATUS=$(echo "$RESP" | tail -1)
BODY=$(echo "$RESP" | head -1)
check_status "TC-01 health" 200 "$STATUS"
check_json_field "TC-01" "$BODY" ".status" "ok"
check_json_field "TC-01" "$BODY" ".dim" "512"
log_info "detector: $(echo $BODY | jq -r '.detector')"
log_info "stored_vectors: $(echo $BODY | jq -r '.stored_vectors')"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  TC-02  Luồng 2 bước: Enroll → Match (cùng người)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# --- Bước 1: Enroll ---
RESP=$(curl_post -X POST \
  -F "file=@${TEST_IMG_A};type=image/jpeg" \
  -F "label=person_a" \
  "$BASE_URL/v1/faces")
STATUS=$(echo "$RESP" | tail -1); BODY=$(echo "$RESP" | head -1)
check_status "TC-02a Enroll" 200 "$STATUS"
check_json_field "TC-02a" "$BODY" ".dim" "512"
FACE_ID=$(echo "$BODY" | jq -r '.id')
log_info "Enrolled face_id = $FACE_ID"

# --- Bước 2: Match cùng người ---
RESP=$(curl_post -X POST \
  -F "file=@${TEST_IMG_B};type=image/jpeg" \
  "$BASE_URL/v1/faces/${FACE_ID}/match")
STATUS=$(echo "$RESP" | tail -1); BODY=$(echo "$RESP" | head -1)
check_status "TC-02b Match cùng người" 200 "$STATUS"
check_json_field "TC-02b result" "$BODY" ".result" "Match"
check_json_field "TC-02b match"  "$BODY" ".match"  "1"
SIM=$(echo "$BODY" | jq -r '.similarity')
log_info "Similarity (cùng người) = $SIM  (kỳ vọng > $MATCH_THRESHOLD)"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  TC-03  Match khác người → 0 Match"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
RESP=$(curl_post -X POST \
  -F "file=@${TEST_IMG_C};type=image/jpeg" \
  "$BASE_URL/v1/faces/${FACE_ID}/match")
STATUS=$(echo "$RESP" | tail -1); BODY=$(echo "$RESP" | head -1)
check_status "TC-03 Match khác người" 200 "$STATUS"
check_json_field "TC-03 result" "$BODY" ".result" "0 Match"
check_json_field "TC-03 match"  "$BODY" ".match"  "0"
SIM_DIFF=$(echo "$BODY" | jq -r '.similarity')
log_info "Similarity (khác người) = $SIM_DIFF  (kỳ vọng < $MATCH_THRESHOLD)"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  TC-04  /v1/compare – so trực tiếp 2 ảnh"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# TC-04a: Cùng người
RESP=$(curl_post -X POST \
  -F "file_a=@${TEST_IMG_A};type=image/jpeg" \
  -F "file_b=@${TEST_IMG_B};type=image/jpeg" \
  "$BASE_URL/v1/compare")
STATUS=$(echo "$RESP" | tail -1); BODY=$(echo "$RESP" | head -1)
check_status "TC-04a compare cùng người" 200 "$STATUS"
check_json_field "TC-04a match" "$BODY" ".match" "1"

# TC-04b: Khác người
RESP=$(curl_post -X POST \
  -F "file_a=@${TEST_IMG_A};type=image/jpeg" \
  -F "file_b=@${TEST_IMG_C};type=image/jpeg" \
  "$BASE_URL/v1/compare")
STATUS=$(echo "$RESP" | tail -1); BODY=$(echo "$RESP" | head -1)
check_status "TC-04b compare khác người" 200 "$STATUS"
check_json_field "TC-04b match" "$BODY" ".match" "0"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  TC-05  /v1/embeddings – lấy vector, không lưu"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
RESP=$(curl_post -X POST \
  -F "file=@${TEST_IMG_A};type=image/jpeg" \
  "$BASE_URL/v1/embeddings")
STATUS=$(echo "$RESP" | tail -1); BODY=$(echo "$RESP" | head -1)
check_status "TC-05 embeddings" 200 "$STATUS"
check_json_field "TC-05 dim" "$BODY" ".dim" "512"
VEC_LEN=$(echo "$BODY" | jq '.embedding | length')
log_info "Embedding length = $VEC_LEN  (kỳ vọng 512)"
[ "$VEC_LEN" -eq 512 ] && log_pass "TC-05 embedding length = 512" || log_fail "TC-05 embedding length ≠ 512"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  TC-06  /v1/identify – 1:N"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# Enroll thêm person_b
curl_post -X POST \
  -F "file=@${TEST_IMG_C};type=image/jpeg" \
  -F "label=person_b" \
  "$BASE_URL/v1/faces" > /dev/null 2>&1

RESP=$(curl_post -X POST \
  -F "file=@${TEST_IMG_A};type=image/jpeg" \
  -F "top_k=5" \
  "$BASE_URL/v1/identify")
STATUS=$(echo "$RESP" | tail -1); BODY=$(echo "$RESP" | head -1)
check_status "TC-06 identify" 200 "$STATUS"
TOP_LABEL=$(echo "$BODY" | jq -r '.candidates[0].label')
log_info "Top candidate label = $TOP_LABEL  (kỳ vọng person_a)"
[ "$TOP_LABEL" = "person_a" ] && log_pass "TC-06 top candidate đúng người" || log_fail "TC-06 top candidate sai người (nhận '$TOP_LABEL')"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  TC-07  Lỗi 404 – face_id không tồn tại"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
RESP=$(curl_post -X POST \
  -F "file=@${TEST_IMG_A};type=image/jpeg" \
  "$BASE_URL/v1/faces/non-existent-id-99999/match")
STATUS=$(echo "$RESP" | tail -1)
check_status "TC-07 404 không tìm thấy face_id" 404 "$STATUS"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  TC-08  Lỗi 422 – không phát hiện mặt"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
RESP=$(curl_post -X POST \
  -F "file=@${BLANK_IMG};type=image/jpeg" \
  "$BASE_URL/v1/faces")
STATUS=$(echo "$RESP" | tail -1)
check_status "TC-08 422 không thấy mặt" 422 "$STATUS"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  TC-09  Lỗi 400 – định dạng file không hỗ trợ"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
RESP=$(curl_post -X POST \
  -F "file=@${FAKE_TXT};type=text/plain" \
  "$BASE_URL/v1/faces")
STATUS=$(echo "$RESP" | tail -1)
check_status "TC-09 400 định dạng không hỗ trợ" 400 "$STATUS"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  TC-10  Lỗi 400 – file rỗng"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
EMPTY="tests/assets/empty.jpg"
: > "$EMPTY"   # Tạo file trống
RESP=$(curl_post -X POST \
  -F "file=@${EMPTY};type=image/jpeg" \
  "$BASE_URL/v1/faces")
STATUS=$(echo "$RESP" | tail -1)
check_status "TC-10 400 file rỗng" 400 "$STATUS"
rm -f "$EMPTY"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  TC-11  Lỗi 413 – ảnh vượt giới hạn kích thước"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
BIG_IMG="tests/assets/big_image.jpg"
python3 -c "
import cv2, numpy as np
img = np.random.randint(0, 256, (5000, 5000, 3), dtype=np.uint8)
cv2.imwrite('$BIG_IMG', img, [cv2.IMWRITE_JPEG_QUALITY, 100])
print('Big image size:', __import__('os').path.getsize('$BIG_IMG') // 1024 // 1024, 'MB')
" 2>/dev/null || log_skip "TC-11 bỏ qua (không tạo được ảnh lớn)"
if [ -f "$BIG_IMG" ]; then
  IMG_MB=$(python3 -c "import os; print(os.path.getsize('$BIG_IMG')/1024/1024)" 2>/dev/null)
  log_info "Kích thước ảnh test = ${IMG_MB}MB"
  RESP=$(curl_post -X POST \
    -F "file=@${BIG_IMG};type=image/jpeg" \
    "$BASE_URL/v1/faces")
  STATUS=$(echo "$RESP" | tail -1)
  check_status "TC-11 413 ảnh quá lớn" 413 "$STATUS"
  rm -f "$BIG_IMG"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  TC-12  GET /v1/faces – danh sách"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
RESP=$(curl -s -w '\n%{http_code}' $([ -n "$API_KEY" ] && echo "-H 'X-API-Key: $API_KEY'") \
  "$BASE_URL/v1/faces?limit=10&offset=0")
STATUS=$(echo "$RESP" | tail -1); BODY=$(echo "$RESP" | head -1)
check_status "TC-12 GET /v1/faces" 200 "$STATUS"
COUNT=$(echo "$BODY" | jq 'length')
log_info "Số bản ghi hiện tại = $COUNT"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  TC-13  DELETE /v1/faces/{id}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# Enroll rồi xóa
RESP=$(curl_post -X POST \
  -F "file=@${TEST_IMG_A};type=image/jpeg" \
  -F "label=to_delete" \
  "$BASE_URL/v1/faces")
DEL_ID=$(echo "$RESP" | head -1 | jq -r '.id')
RESP=$(curl -s -w '\n%{http_code}' $([ -n "$API_KEY" ] && echo "-H 'X-API-Key: $API_KEY'") \
  -X DELETE "$BASE_URL/v1/faces/${DEL_ID}")
STATUS=$(echo "$RESP" | tail -1); BODY=$(echo "$RESP" | head -1)
check_status "TC-13 DELETE /v1/faces/{id}" 200 "$STATUS"
check_json_field "TC-13 deleted" "$BODY" ".deleted" "$DEL_ID"
# Xóa lần 2 → 404
RESP=$(curl -s -w '\n%{http_code}' $([ -n "$API_KEY" ] && echo "-H 'X-API-Key: $API_KEY'") \
  -X DELETE "$BASE_URL/v1/faces/${DEL_ID}")
STATUS=$(echo "$RESP" | tail -1)
check_status "TC-13b DELETE lần 2 → 404" 404 "$STATUS"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  TC-14  API Key – 401 khi sai key"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# Chỉ chạy khi service đang bật API_KEY
HEALTH=$(curl -s "$BASE_URL/health")
# Thử gọi với key sai
RESP=$(curl -s -w '\n%{http_code}' \
  -H "X-API-Key: wrong-key-123" \
  -X POST -F "file=@${TEST_IMG_A};type=image/jpeg" \
  "$BASE_URL/v1/faces")
STATUS=$(echo "$RESP" | tail -1)
if [ -n "$API_KEY" ]; then
  check_status "TC-14 401 sai API key" 401 "$STATUS"
else
  log_skip "TC-14 bỏ qua (API_KEY không được cấu hình – xem config API_KEY)"
  log_info "  Để test: đặt API_KEY=your-secret trong docker-compose.yml và chạy lại"
fi

# =============================================================================
# Tóm tắt kết quả
# =============================================================================
echo ""
echo "=============================================="
echo "  KẾT QUẢ"
echo "  PASS: $PASS  |  FAIL: $FAIL  |  SKIP: $SKIP"
echo "=============================================="
[ $FAIL -eq 0 ] && echo -e "${GREEN}  ✅ Tất cả test passed!${NC}" || echo -e "${RED}  ❌ Có $FAIL test thất bại.${NC}"
echo ""
