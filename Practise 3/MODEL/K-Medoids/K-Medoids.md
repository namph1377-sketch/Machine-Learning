# BÁO CÁO TRIỂN KHAI MÔ HÌNH PHÂN CỤM K-MEDOIDS (PAM) FROM SCRATCH

Notebook: [`K-Medoids_Pam.ipynb`](K-Medoids_Pam.ipynb)

## 1. Giới thiệu bài toán

Trong CRM và Marketing dựa trên dữ liệu, **Phân khúc khách hàng (Customer Segmentation)** là bài toán kinh điển thuộc Học không giám sát.

Bài toán khai phá cấu trúc tiềm ẩn từ tập **Customer Personality Analysis** — nhân khẩu học (Tuổi, Thu nhập, Hôn nhân, Số con) và hành vi (Chi tiêu, Kênh mua, Thời gian gắn bó, Phản hồi chiến dịch). Mục tiêu: gom nhóm khách hàng tương đồng để tối ưu chăm sóc và cá nhân hóa marketing.

Notebook triển khai **K-Medoids (PAM)** from scratch: chọn **medoid** là một khách hàng thực làm đại diện cụm — robust hơn với outlier và dễ diễn giải nghiệp vụ hơn centroid K-Means.

---

## 2. Mục lục triển khai

1. [Yêu cầu Dữ liệu đầu vào & Thư viện hỗ trợ](#3-yêu-cầu-dữ-liệu-đầu-vào--thư-viện-hỗ-trợ)
2. [Cơ sở Lý thuyết K-Medoids (PAM)](#4-cơ-sở-lý-thuyết-k-medoids-pam)
3. [Cải tiến Khởi tạo: K-Medoids++](#5-cải-tiến-khởi-tạo-k-medoids)
4. [Cấu trúc Mã nguồn (Class & Function)](#6-cấu-trúc-mã-nguồn-class--function)
5. [Kiểm định và Tối ưu hóa Tham số K](#7-kiểm-định-và-tối-ưu-hóa-tham-số-k)
6. [Đọc kết quả & Đặc tả Phân khúc](#8-đọc-kết-quả--đặc-tả-phân-khúc)

**Pipeline notebook:**

| Giai đoạn | Nội dung |
|-----------|----------|
| 1 | Đọc dữ liệu, PCA 2D (chỉ visualize) |
| 2 | Cài đặt PAM + Silhouette from scratch |
| 3 | Quét K = 2…10 → Elbow + Silhouette → chọn **K = 4** |
| 4 | Profile cụm, medoid, xuất CSV, scatter PCA |

---

## 3. Yêu cầu Dữ liệu đầu vào & Thư viện hỗ trợ

### 3.1. Dữ liệu đầu vào (`customer_personality_preprocessed.csv`)

Tiền xử lý do pipeline riêng đảm nhiệm. Mô hình yêu cầu:

* **Kích thước:** 2.237 mẫu × 24 cột số.
* **Đã chuẩn hóa (Z-score):** Giá trị profile cụm là độ lệch so với trung bình, không phải đơn vị gốc.
* **Số hóa hoàn toàn:** Biến định tính đã mã hóa; không chứa NaN.
* **PCA chỉ để visualize:** Clustering chạy trên 24 đặc trưng gốc; PCA 2D dùng cho scatter plot.

```python
FEATURE_COLS = [c for c in df.columns if c != 'Cluster']
X = df[FEATURE_COLS].values
```

### 3.2. Đầu ra

* **CSV:** `customer_personality_clustered_v2.csv` (2.237 × 25) — thêm cột `Cluster`.
* **Mô hình:** `labels`, `medoid_indices_`, `medoids`, `total_cost_`, `n_iters_`.
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

1. **Khởi tạo:** Chọn $K$ medoid (K-Medoids++).
2. **Gán cụm:** Tính khoảng cách Euclidean, gán mỗi điểm vào medoid gần nhất:

$$
d(X_i,\, M_j) = \sqrt{\sum_{m=1}^{p} \left(X_{im} - M_{jm}\right)^2}
$$

3. **Hoán đổi (Swap):** Thử thay medoid $M_i$ bằng điểm $X_j$; chấp nhận swap **giảm cost nhiều nhất**.
4. **Hội tụ:** Dừng khi không còn swap cải thiện, hoặc đạt `max_iters=100`.

**Total Cost (hàm mục tiêu):**

$$
J = \sum_{i=1}^{n} \min_{j=1,\ldots,k} d(X_i,\, M_j)
$$

**Silhouette (from scratch):**

$$
s(i) = \frac{b(i) - a(i)}{\max\bigl(a(i),\, b(i)\bigr)}
$$

- $a(i)$: khoảng cách trung bình từ điểm $i$ đến các điểm cùng cụm.
- $b(i)$: khoảng cách trung bình từ điểm $i$ đến cụm gần nhất khác.
- Điểm Silhouette = trung bình $s(i)$ trên toàn tập, $\in [-1,\, 1]$.
- Ma trận khoảng cách $D \in \mathbb{R}^{n \times n}$ tiền tính một lần, tái sử dụng khi quét $K$.

---

## 5. Cải tiến Khởi tạo: K-Medoids++

Khởi tạo ngẫu nhiên thuần có rủi ro hội tụ cục bộ. Notebook dùng **K-Medoids++** — các medoid ban đầu càng xa nhau càng tốt:

1. Chọn medoid đầu tiên ngẫu nhiên.
2. Với mỗi điểm $X_i$ chưa được chọn, tính khoảng cách ngắn nhất tới tập medoid hiện có:

$$
D(X_i) = \min_{M \in \mathcal{M}} d(X_i,\, M)
$$

3. Chọn medoid tiếp theo theo xác suất tỷ lệ với $D(X_i)$:

$$
P(X_i) \propto D(X_i)
$$

4. Lặp đến đủ $K$ medoid (`random_state=42`).

> Medoid luôn là khách hàng thực → `df.iloc[medoid_indices_]` là chân dung điển hình từng cụm.

---

## 6. Cấu trúc Mã nguồn (Class & Function)

### 6.1. Lớp `KMedoidsFromScratch`

**`__init__`:** `n_clusters`, `max_iters=100`, `random_state=42`, `verbose=False`

**Sau `fit`:** `medoids`, `medoid_indices_`, `labels`, `total_cost_`, `n_iters_`

**Private:** `_init_medoids_plusplus`, `_assign_clusters_with_medoid_indices`, `_compute_cost`, `_find_best_swap`

**Public:** `fit(X, D=None)` — vòng lặp PAM (tái dùng ma trận `D`); `predict(X)` — gán nhãn dữ liệu mới theo `self.medoids`

### 6.2. Hàm hỗ trợ

* `compute_pairwise_distance_matrix(X)` — ma trận Euclidean (n×n)
* `silhouette_score_from_scratch(X, labels, D=None)` — đánh giá, không train
* `pca_project` / `pca_transform` — PCA from scratch, chỉ visualize

---

## 7. Kiểm định và Tối ưu hóa Tham số K

Quét $K \in \{2,\, 3,\, \ldots,\, 10\}$, `max_iters=100`, `random_state=42`. Kết quả chính:

| K | Silhouette ↑ | Total Cost ↓ | PAM passes |
|---:|---:|---:|---:|
| 2 | **0.2274** | 9560.04 | 3 |
| 3 | 0.0700 | 9276.73 | 5 |
| 4 | 0.0692 | **9003.53** | 6 |
| 5 | 0.0552 | 8762.02 | 8 |
| 6 | 0.0673 | 8525.61 | 12 |
| 7 | 0.0643 | 8394.72 | 10 |
| 8 | 0.0652 | 8212.92 | 11 |
| 9 | 0.0714 | 8073.58 | 10 |
| 10 | 0.0748 | 7939.30 | 12 |

### 7.1. Elbow (Total Cost)

Cost giảm mạnh K=2→3 (−283) và K=3→4 (−273). Từ K=5 trở đi vẫn giảm nhưng chậm dần → khuỷu tay quanh **K=3–4**. Tại K=4, cost giảm ~5,8% so với K=2.

### 7.2. Silhouette

K=2 đỉnh (**0.2274**). Từ K=3, Silhouette ~0,07 và dao động 0,055–0,075 — cụm chồng lấn tăng so với K=2. K=4 (0,0692) gần tương đương K=3 (0,0700).

> **Trade-off:** K=2 tốt nhất về toán học; notebook chọn **`optimal_k = 4`** để có 4 phân khúc hành động được. Có thể đổi K tại cell `optimal_k` mà không train lại (`models_k`).

---

## 8. Đọc kết quả & Đặc tả Phân khúc

Sau khi chọn K: `value_counts` → `cluster_summary = df.groupby('Cluster').mean()` → trích medoid → xuất CSV → scatter PCA.

**Mô hình cuối:** K=4 | Silhouette 0.0692 | Cost 9003.53 | 6 pass PAM

| Cụm | Số lượng | Tỷ lệ | `medoid_indices_` |
|-----|----------|-------|-------------------|
| 0 | 492 | 22.0% | 898 |
| 1 | 328 | 14.7% | 795 |
| 2 | 830 | 37.1% | 82 |
| 3 | 587 | 26.2% | 2235 |

Phân bố **lệch** — C2 chiếm 37,1% (lớn nhất), C1 nhỏ nhất 14,7%.

### 8.1. K = 2 (góc nhìn tổng quan — tối ưu toán học)

Silhouette **0.2274** — hai miền tách bạch nhất. Hạn chế: chỉ 2 nhóm đối lập, quá thô cho nhiều chiến lược marketing.

### 8.2. K = 4 (phương án được chọn)

#### Bảng trung tâm cụm (đã chuẩn hóa)

| Chỉ số | C0 | C1 | C2 | C3 |
| :--- | :---: | :---: | :---: | :---: |
| **Income** | −0.35 | **+0.95** | −0.65 | +0.68 |
| **Age** | −0.04 | −0.32 | −0.21 | **+0.51** |
| **Children** | +0.34 | **−1.01** | +0.41 | −0.30 |
| **TotalSpending** | −2.64 | **+7.63** | −3.23 | +2.51 |
| **MntMeatProducts** | −0.50 | **+1.57** | −0.59 | +0.38 |
| **NumCatalogPurchases** | −0.49 | **+1.11** | −0.66 | +0.73 |
| **NumWebVisitsMonth** | +0.34 | **−0.94** | **+0.57** | −0.57 |
| **Response** | −0.07 | **+0.41** | −0.16 | +0.05 |

*(Dương = cao hơn TB toàn tập; âm = thấp hơn)*

#### Chân dung phân khúc

**C1 — VIP / Chi tiêu mạnh** · 328 KH (15%)

Thu nhập cao nhất (+0.95), chi tiêu vượt trội (+7.63), ít con (−1.01). Mua chủ yếu Catalog (+1.11) & Store (+0.87), ít lướt Web (−0.94). Phản hồi chiến dịch tích cực (Response +0.41, AcceptedCmp5 +0.65).

→ *Chiến lược:* loyalty cao cấp, upsell premium (rượu, thịt), ưu đãi qua Catalog/Store; tránh spam quảng cáo số.

**C2 — Gia đình ngân sách thấp** · 830 KH (37%)

Cụm lớn nhất: thu nhập thấp (−0.65), chi tiêu thấp toàn danh mục, nhiều con (+0.41), hôn nhân cao (+0.65). Lướt Web nhiều (+0.57) nhưng mua ít mọi kênh — "xem nhiều, mua ít".

→ *Chiến lược:* combo gia đình giá mềm, khuyến mãi theo mùa; ưu tiên ngân sách do tỷ trọng lớn.

**C0 — Thụ động / Ít tương tác** · 492 KH (22%)

Thu nhập và chi tiêu đều thấp, hôn nhân khác biệt (−1.23), có con (+0.34). Lướt Web (+0.34) nhưng mua kênh thấp; phản hồi chiến dịch âm.

→ *Chiến lược:* khảo sát nhu cầu, ưu đãi kích hoạt lần mua đầu, remarketing nhẹ.

**C3 — Trung-cao tuổi / Đa kênh** · 587 KH (26%)

Thu nhập khá (+0.68), tuổi cao (+0.51), học vấn cao (+0.33). Mua Web (+0.64), Catalog (+0.73), Store (+0.84) đều cao; ít duyệt Web (−0.57).

→ *Chiến lược:* catalogue marketing, ưu đãi in-store, chiến dịch cross-channel.

### 8.3. Đánh giá & Khuyến nghị

| Tiêu chí | K = 2 | K = 3 | K = 4 (chọn) |
|----------|-------|-------|--------------|
| Silhouette | **0.2274** | 0.0700 | 0.0692 |
| Total Cost | 9560 | 9277 | **9004** |
| Độ chi tiết | Quá thô | 3 nhóm | **4 nhóm** |
| Hành động | 2 chiến dịch | 3 chiến lược | **4 chiến lược** |

* **Toán học:** K=2 cho Silhouette tốt nhất; K=3 và K=4 tương đương (~0,07).
* **Nghiệp vụ:** K=4 tách VIP (C1), gia đình ngân sách thấp (C2), thụ động (C0), trung-cao tuổi đa kênh (C3) — medoid là KH thực, dễ trình bày stakeholder.

**Trực quan hóa:** Elbow + Silhouette (K=2→10); scatter PCA 2D — màu theo cụm, medoid marker X đen.

**Khuyến nghị:** Dùng kết quả **K = 4** (`customer_personality_clustered_v2.csv`) làm cơ sở triển khai Marketing Automation.