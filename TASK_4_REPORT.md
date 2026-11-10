# Task 4: Dataset Creation & System Evaluation Report

## 1. Thong tin Dataset thu nghiem
- **Tong so anh quet:** 83 anh
- **So anh doc hop le:** 80 anh (loai bo 3 anh hong/khong doc duoc)
- **So luong doi tuong:** 15 nguoi (`Person1` den `Person15`)
- **Dac diem du lieu:** Anh chup o nhieu goc do va dieu kien khac nhau (chinh dien, cuoi, nghieng trai/phai, deo kinh, anh toi/mo, anh trang den, anh lat/xoay).

## 2. Ket qua Danh gia Mo hinh (ArcFace 512-d + RetinaFace)
Chay script `scripts/evaluate.py` tren tap dataset thuc te:
- **Tong so cap so sanh:** 3,160 cap
  - **Cap cung nguoi (Genuine):** 197 cap
  - **Cap khac nguoi (Impostor):** 2,963 cap
- **Phan phoi diem tuong dong (Cosine Similarity):**
  - **Cung nguoi:** Mean = `0.668`, Min = `0.462`
  - **Khac nguoi:** Mean = `0.012`, Max = `0.262`

### Thong ke hieu nang voi Nguong (Threshold) = 0.40:
- **Do chinh xac (Accuracy):** 1.000 (100%%)
- **Ty le nhan dien sai (FAR - False Acceptance Rate):** 0.000 (0.0%%)
- **Ty le tu choi sai (FRR - False Rejection Rate):** 0.000 (0.0%%)
- **True Positives (TP):** 197 | **False Positives (FP):** 0
- **True Negatives (TN):** 2963 | **False Negatives (FN):** 0

## 3. De xuat Nguong Hoat dong (Threshold Proposal)
- **Nguong goi y tu toan hoc:** `0.27` (dat Accuracy 1.000).
- **Nguong chot de xuat cho he thong:** **`MATCH_THRESHOLD = 0.40`**
  - *Ly do:* Khoang diem khac nguoi cao nhat la `0.262` va cung nguoi thap nhat la `0.462`. Muc `0.40` nam o trung tam khoang an toan (Safety Margin), giup chong nhan dien nham tuyet doi khi trien khai thuc te.

## 4. Tep Ban giao (Deliverables)
- Du lieu ma tran chi tiet da luu trong thu muc `report/`:
  - `report/pairs.csv`: Chi tiet ket qua so sanh cua tung cap anh.
  - `report/similarity_matrix.csv`: Ma tran diem cosine similarity giua 80 anh.
