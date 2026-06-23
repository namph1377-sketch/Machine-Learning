# BÁO CÁO TRIỂN KHAI MÔ HÌNH PHÂN CỤM K-MEDOIDS (PAM) FROM SCRATCH

## 1. Giới thiệu bài toán

Trong CRM và Marketing dựa trên dữ liệu, **Phân khúc khách hàng (Customer Segmentation)** là bài toán kinh điển thuộc Học không giám sát.

Bài toán khai phá cấu trúc tiềm ẩn từ tập **Customer Personality Analysis** — nhân khẩu học (Tuổi, Thu nhập, Hôn nhân, Số con) và hành vi (Chi tiêu, Kênh mua, Thời gian gắn bó, Phản hồi chiến dịch). Mục tiêu: gom nhóm khách hàng tương đồng để tối ưu chăm sóc và cá nhân hóa marketing.

Khác K-Means dùng **centroid** (trung bình, có thể không thuộc tập dữ liệu), **K-Medoids** chọn **medoid** — một khách hàng thực làm đại diện cụm. Notebook triển khai **PAM (Partitioning Around Medoids)** với hoán đổi medoid, robust hơn với outlier và dễ diễn giải nghiệp vụ.

---

## 2. Mục lục triển khai

1. [Yêu cầu Dữ liệu đầu vào & Thư viện hỗ trợ](#3-yêu-cầu-dữ-liệu-đầu-vào--thư-viện-hỗ-trợ)
2. [Cơ sở Lý thuyết K-Medoids (PAM)](#4-cơ-sở-lý-thuyết-k-medoids-pam)
3. [Cải tiến Khởi tạo: K-Medoids++](#5-cải-tiến-khởi-tạo-k-medoids)
4. [Cấu trúc Mã nguồn (Class & Function)](#6-cấu-trúc-mã-nguồn-class--function)
5. [Kiểm định và Tối ưu hóa Tham số K](#7-kiểm-định-và-tối-ưu-hóa-tham-số-k)
6. [Đọc kết quả & Đặc tả Phân khúc](#8-đọc-kết-quả--đặc-tả-phân-khúc)

---

## 3. Yêu cầu Dữ liệu đầu vào & Thư viện hỗ trợ

### 3.1. Dữ liệu đầu vào (`customer_personality_preprocessed.csv`)

Tiền xử lý do pipeline riêng đảm nhiệm. Mô hình yêu cầu:

* **Kích thước:** 2.237 mẫu × 24 cột số (`X = df.values`).
* **Đã chuẩn hóa (Z-score):** Giá trị profile cụm là độ lệch so với trung bình, không phải đơn vị gốc.
* **Số hóa hoàn toàn:** Biến định tính đã mã hóa; không chứa NaN.
* **PCA chỉ để visualize:** Clustering chạy trên 24 đặc trưng gốc; PCA 2D dùng cho scatter plot.

### 3.2. Đầu ra

* **CSV:** `customer_personality_clustered_v2.csv` (2.237 × 25) — thêm cột `Cluster`.
* **Mô hình:** `labels`, `medoid_indices_`, `medoids`, `total_cost_`.
* **Biểu đồ:** Elbow + Silhouette (K=2→10); scatter PCA 2D kèm medoid (marker X).

### 3.3. Thư viện

Không dùng `sklearn.cluster` cho lõi phân cụm:

* `numpy` — khoảng cách, PAM, PCA, Silhouette from scratch
* `pandas` — đọc/ghi CSV, `groupby`, `value_counts`
* `matplotlib.pyplot` & `seaborn` — Elbow, Silhouette, scatter

---

## 4. Cơ sở Lý thuyết K-Medoids (PAM)

K-Medoids tối thiểu hóa **Total Cost** (TDAD) — tổng khoảng cách Euclidean từ mỗi điểm đến medoid cụm của nó.

| Tiêu chí | K-Means | K-Medoids (PAM) |
|----------|---------|-----------------|
| Đại diện cụm | Centroid (trung bình) | Medoid (điểm dữ liệu thực) |
| Hàm mục tiêu | WCSS / Inertia | Total Cost |
| Cập nhật | Trung bình các điểm | Hoán đổi medoid nếu Cost giảm |
| Outlier | Nhạy cảm | Bền hơn |

**Quy trình 4 bước lặp:**

1. **Khởi tạo:** Chọn K medoid (K-Medoids++).
2. **Gán cụm:** `d(X_i, M_j) = sqrt( sum( (X_im - M_jm)^2 ) )` → gán vào medoid gần nhất.
3. **Hoán đổi (Swap):** Thử thay medoid $M_i$ bằng điểm $X_j$; chấp nhận nếu Total Cost giảm.
4. **Hội tụ:** Dừng khi không còn swap cải thiện, hoặc đạt `max_iters`.

**Total Cost:** `J = sum_i min_j d(X_i, M_j)`

**Silhouette (from scratch):** `s(i) = (b(i) - a(i)) / max(a(i), b(i))` — trung bình trên toàn tập, ∈ [-1, 1]. Ma trận khoảng cách $D$ (n×n) tiền tính một lần, tái sử dụng khi quét K.

---

## 5. Cải tiến Khởi tạo: K-Medoids++

Khởi tạo ngẫu nhiên thuần có rủi ro hội tụ cục bộ. Notebook dùng **K-Medoids++** — các medoid ban đầu càng xa nhau càng tốt:

1. Chọn medoid đầu tiên ngẫu nhiên.
2. Tính $D(X_i)$ = khoảng cách ngắn nhất tới medoid đã chọn.
3. Chọn medoid tiếp theo với $P(X_i) \propto D(X_i)$ trên các điểm chưa chọn.
4. Lặp đến đủ K medoid (`random_state=42`).

> Medoid luôn là khách hàng thực → `df.iloc[medoid_indices_]` là chân dung điển hình từng cụm.

---

## 6. Cấu trúc Mã nguồn (Class & Function)

### 6.1. Lớp `KMedoidsFromScratch`

**`__init__`:** `n_clusters`, `max_iters=100`, `tol=1e-4`, `random_state=42`

**Sau `fit`:** `medoids`, `medoid_indices_`, `labels`, `total_cost_`

**Private:** `_init_medoids_plusplus`, `_assign_clusters`, `_assign_clusters_with_medoids`, `_compute_cost`, `_compute_total_cost`

**Public:** `fit(X)` — vòng lặp PAM; `predict(X)` — gán nhãn dữ liệu mới

### 6.2. Hàm hỗ trợ

* `compute_pairwise_distance_matrix(X)` — ma trận Euclidean (n×n)
* `silhouette_score_from_scratch(X, labels, D=None)` — đánh giá, không train
* `pca_project` / `pca_transform` — PCA from scratch, chỉ visualize

---

## 7. Kiểm định và Tối ưu hóa Tham số K

Quét **K ∈ {2, …, 10}**, `max_iters=100`, `random_state=42`. Kết quả chính:

| K | Silhouette ↑ | Total Cost ↓ |
|---|-------------|--------------|
| 2 | **0.2274** | 9560.04 |
| 3 | 0.1541 | 9173.91 |
| 4 | 0.0628 | **8909.62** |
| 5 | 0.0562 | 8746.22 |
| 6–10 | 0.06–0.08 | Tiếp tục giảm chậm |

### 7.1. Elbow (Total Cost)

Cost giảm mạnh K=2→3 (−386) và K=3→4 (−264). Từ K=5 trở đi giảm chậm → khuỷu tay quanh **K=3–4**.

### 7.2. Silhouette

K=2 đỉnh (**0.2274**), K=3 còn chấp nhận được (0.1541). K≥4 suy giảm mạnh — cụm chồng lấn tăng.

> **Trade-off:** K=2/3 tốt về toán học; notebook chọn **`optimal_k = 4`** để có 4 phân khúc hành động được. Có thể đổi K trong Cell 11 mà không train lại (`models_k`).

---

## 8. Đọc kết quả & Đặc tả Phân khúc

Sau khi chọn K: `value_counts` → `groupby('Cluster').mean()` → trích medoid → xuất CSV → scatter PCA.

**Mô hình cuối:** K=4 | Silhouette 0.0628 | Cost 8909.62

| Cụm | Số lượng | Tỷ lệ | `medoid_indices_` |
|-----|----------|-------|-------------------|
| 0 | 623 | 27.8% | 234 |
| 1 | 599 | 26.8% | 841 |
| 2 | 603 | 27.0% | 766 |
| 3 | 412 | 18.4% | 364 |

### 8.1. K = 2 (góc nhìn tổng quan — tối ưu toán học)

Silhouette **0.2274** — hai miền tách bạch nhất. Hạn chế: chỉ 2 nhóm đối lập, quá thô cho nhiều chiến lược marketing.

### 8.2. K = 4 (phương án được chọn)

#### Bảng trung tâm cụm (đã chuẩn hóa)

| Chỉ số | C0 | C1 | C2 | C3 |
| :--- | :---: | :---: | :---: | :---: |
| **Income** | −0.68 | **+0.97** | +0.21 | −0.68 |
| **Age** | −0.27 | +0.07 | **+0.44** | −0.32 |
| **TotalSpending** | **−3.60** | **+5.89** | +0.06 | −3.22 |
| **NumStorePurchases** | −0.77 | **+0.79** | +0.52 | −0.75 |
| **NumWebPurchases** | −0.74 | +0.37 | **+0.81** | −0.60 |
| **DaysCustomer** | −0.46 | −0.01 | +0.32 | +0.25 |

#### Chân dung phân khúc

* **C1 — VIP (599 KH, 26.8%):** Thu nhập + chi tiêu cao nhất; ưu tiên mua tại cửa hàng. → Upsell cao cấp, ưu đãi VIP.
* **C2 — Trung bình / Digital (603 KH, 27.0%):** Chi tiêu trung tính, tuổi cao hơn, thiên online, gắn bó lâu. → Marketing digital, loyalty, combo.
* **C0 — Tiết kiệm A (623 KH, 27.8%):** Thu nhập thấp, chi tiêu rất thấp, ít tương tác, gắn bó ngắn. → Ưu đãi entry-level, tăng tần suất ghé thăm.
* **C3 — Tiết kiệm B (412 KH, 18.4%):** Tương tự C0 về thu nhập/chi tiêu nhưng gắn bó lâu hơn. → Giữ chân KH trung thành, khuyến mãi theo lịch sử mua.

### 8.3. Đánh giá & Khuyến nghị

| Tiêu chí | K = 2 | K = 3 | K = 4 (chọn) |
|----------|-------|-------|--------------|
| Silhouette | **0.2274** | 0.1541 | 0.0628 |
| Total Cost | 9560 | 9174 | **8910** |
| Độ chi tiết | Quá thô | 3 nhóm | **4 nhóm** |
| Hành động | 2 chiến dịch | 3 chiến lược | **4 chiến lược** |

* **Toán học:** K=2 (hoặc K=3) cho Silhouette tốt hơn.
* **Nghiệp vụ:** K=4 tách VIP, Mid/Digital và hai nhóm tiết kiệm khác hành vi gắn bó — medoid là KH thực, dễ trình bày stakeholder.

**Khuyến nghị:** Dùng kết quả **K = 4** (`customer_personality_clustered_v2.csv`) làm cơ sở triển khai Marketing Automation.