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

* **Kích thước:** 2.237 mẫu × 24 cột số.
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
2. **Gán cụm:** Tính khoảng cách Euclidean, gán mỗi điểm vào medoid gần nhất:

$$
d(X_i,\, M_j) = \sqrt{\sum_{m=1}^{p} \left(X_{im} - M_{jm}\right)^2}
$$

3. **Hoán đổi (Swap):** Thử thay medoid $M_i$ bằng điểm $X_j$; chấp nhận nếu Total Cost giảm.
4. **Hội tụ:** Dừng khi không còn swap cải thiện, hoặc đạt `max_iters`.

**Total Cost (hàm mục tiêu):**

$$
J = \sum_{i=1}^{n} \min_{j=1,\ldots,k} d(X_i,\, M_j)
$$

**Silhouette (from scratch):**

$$
s(i) = \frac{b(i) - a(i)}{\max\bigl(a(i),\, b(i)\bigr)}
$$

- $a(i)$: khoảng cách trung bình từ điểm $i$ đến các điểm cùng cụm.
- $b(i)$: khoảng cách trung bình từ điểm $i$ đến một cụm gần nhất khác.
- Điểm Silhouette = trung bình $s(i)$ trên toàn tập, $\in [-1,\, 1]$.
- Ma trận khoảng cách $D \in \mathbb{R}^{n \times n}$ tiền tính một lần, tái sử dụng khi quét K.

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

Quét $K \in \{2,\, 3,\, \ldots,\, 10\}$, `max_iters=100`, `random_state=42`. Kết quả chính:

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

K=2 đỉnh (**0.2274**), K=3 còn chấp nhận được (0.1541). $K \geq 4$ suy giảm mạnh — cụm chồng lấn tăng.

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

### 8.2. K = 4 (phương án được chọn — Góc nhìn Chi tiết & Cá nhân hóa)

Khi mở rộng lên $K = 4$, nhóm khách hàng thu nhập thấp (tương ứng cụm tiết kiệm ở $K = 2$) được tách thành **hai phân khúc** có hành vi gắn bó khác nhau (C0 vs C3), trong khi nhóm có khả năng chi tiêu cao được phân rã thành **VIP truyền thống** (C1) và **nhóm trung lưu săn deal trên Web** (C2). Medoid của mỗi cụm là một khách hàng thực — có thể trích `df.iloc[medoid_indices_]` để trình bày chân dung điển hình.

#### Bảng chỉ số trung tâm cụm (K = 4, đã chuẩn hóa)

| Chỉ số đặc trưng | C0 | C1 | C2 | C3 |
| :--- | :---: | :---: | :---: | :---: |
| **Income (Thu nhập)** | −0.68 | **+0.97** | +0.21 | −0.68 |
| **Age (Tuổi)** | −0.27 | +0.06 | **+0.44** | −0.32 |
| **Children (Con cái)** | +0.33 | **−0.94** | +0.38 | +0.31 |
| **TotalSpending (Tổng chi tiêu)** | **−3.59** | **+5.89** | +0.06 | −3.22 |
| **MntWines (Chi rượu)** | −0.78 | **+0.84** | +0.47 | −0.73 |
| **MntMeatProducts (Chi thịt)** | −0.64 | **+1.24** | −0.16 | −0.60 |
| **MntFruits (Chi trái cây)** | −0.53 | **+1.03** | −0.16 | −0.46 |
| **NumStorePurchases (Mua tại Store)** | −0.77 | **+0.79** | +0.52 | −0.75 |
| **NumWebPurchases (Mua qua Web)** | −0.74 | +0.37 | **+0.81** | −0.60 |
| **NumCatalogPurchases (Mua Catalog)** | −0.72 | **+1.14** | +0.06 | −0.66 |
| **NumDealsPurchases (Săn Deal)** | −0.24 | −0.50 | **+0.79** | −0.08 |
| **NumWebVisitsMonth (Ghé Web)** | +0.34 | **−1.06** | +0.28 | **+0.61** |
| **DaysCustomer (Thời gian gắn bó)** | **−0.46** | −0.01 | +0.32 | +0.25 |
| **Response (Phản hồi tổng)** | −0.24 | **+0.31** | −0.01 | −0.07 |
| **AcceptedCmp5 (Chiến dịch 5)** | −0.28 | **+0.64** | −0.15 | −0.28 |

#### Đặc tả chân dung phân khúc

* **C1 — VIP / Thượng lưu Chi tiêu mạnh (Elite High-Spenders) — 599 KH (26.8%)**
    * **Nhân khẩu học:** Thu nhập cao nhất hệ thống (+0.97), gần như không có con nhỏ (−0.94), tuổi trung bình (+0.06). Thời gian gắn bó ở mức trung tính (−0.01).
    * **Hành vi tiêu dùng:** Chi tiêu vượt trội toàn danh mục, đặc biệt Thịt (+1.24), Rượu (+0.84), Trái cây (+1.03). Tổng chi tiêu (+5.89) cao gấp nhiều lần các cụm còn lại. Không phụ thuộc deal (−0.50).
    * **Kênh tương tác:** Ưu tiên mua tại Store (+0.79) và Catalog (+1.14); ghé Web rất ít (−1.06). Phản hồi tích cực với chiến dịch — Campaign 5 (+0.64), Campaign 1 (+0.53), Response (+0.31).
    * **Chiến lược:** Upsell sản phẩm cao cấp (rượu, thịt), dịch vụ VIP, ưu đãi độc quyền qua Catalog/Store. Tránh spam quảng cáo số — nhóm này mua khi có chủ đích, không lướt Web.

* **C2 — Trung lưu / Digital Săn Deal (Web-Centric Mid-Income Bargainers) — 603 KH (27.0%)**
    * **Nhân khẩu học:** Khách hàng lớn tuổi hơn (+0.44), thu nhập khá (+0.21), có con nhỏ (+0.38), gắn bó lâu (+0.32).
    * **Hành vi tiêu dùng:** Chi tiêu trung tính (+0.06) — không vung tay như VIP nhưng cũng không cắt giảm triệt để. Là nhóm **"Vua săn deal"** (+0.79), sẵn sàng mua khi có ưu đãi.
    * **Kênh tương tác:** Số lượng giao dịch qua Web cao nhất (+0.81), mua Store ở mức khá (+0.52). Phản hồi chiến dịch trung tính — cần kích hoạt bằng khuyến mãi cụ thể hơn quảng cáo thương hiệu.
    * **Chiến lược:** Marketing digital, email/push ưu đãi, combo săn deal, chương trình loyalty dài hạn. Đây là nhóm chuyển đổi tốt qua kênh online khi có incentive rõ ràng.

* **C0 — Tiết kiệm A / Gia đình Ngân sách thấp, Gắn bó ngắn (Short-Tenure Budget Families) — 623 KH (27.8%)**
    * **Nhân khẩu học:** Thu nhập dưới trung bình (−0.68), trẻ hơn một chút (−0.27), có con nhỏ (+0.33). **Gắn bó ngắn nhất** (−0.46) — khách hàng mới hoặc ít quay lại.
    * **Hành vi tiêu dùng:** Cắt giảm chi tiêu tối đa — tổng chi tiêu (−3.59) và mọi danh mục (Rượu −0.78, Thịt −0.64) đều âm sâu. Không săn deal (−0.24).
    * **Kênh tương tác:** Mua hàng qua mọi kênh đều thấp (Store −0.77, Web −0.74, Catalog −0.72). Có lướt Web (+0.34) nhưng chưa chuyển đổi. Phản hồi chiến dịch yếu (Response −0.24).
    * **Chiến lược:** Ưu đãi entry-level, tăng tần suất ghé thăm, onboarding cho KH mới. Mục tiêu kéo từ "lướt web" sang "mua lần đầu" trước khi họ rời bỏ.

* **C3 — Tiết kiệm B / Trung thành Ngân sách thấp, Lướt Web (Loyal Budget Browsers) — 412 KH (18.4%)**
    * **Nhân khẩu học:** Tương tự C0 về thu nhập (−0.68) và chi tiêu thấp, nhưng **gắn bó lâu hơn** (+0.25 vs −0.46). Có con nhỏ (+0.31), tuổi trẻ hơn trung bình (−0.32).
    * **Hành vi tiêu dùng:** Tổng chi tiêu vẫn rất thấp (−3.22), toàn bộ danh mục âm. Không phải deal hunter (−0.08) — khác C2 ở chỗ họ không chờ khuyến mãi mà đơn giản là ít mua.
    * **Kênh tương tác:** **Ghé Web thường xuyên nhất trong nhóm tiết kiệm** (+0.61) nhưng mua Web/Store đều thấp — hành vi "xem nhiều, mua ít". Phản hồi chiến dịch gần như bằng 0.
    * **Chiến lược:** Giữ chân KH trung thành lâu năm, retargeting nhẹ nhàng, khuyến mãi theo lịch sử mua. Tận dụng thói quen lướt Web để đẩy sản phẩm giá mềm, gói gia đình — biến "browser" thành "buyer" mà không áp lực quá mức.

### 8.3. Đánh giá & Khuyến nghị

| Tiêu chí | K = 2 | K = 3 | K = 4 (chọn) |
|----------|-------|-------|--------------|
| Silhouette | **0.2274** | 0.1541 | 0.0628 |
| Total Cost | 9560 | 9174 | **8910** |
| Độ chi tiết | Quá thô | 3 nhóm | **4 nhóm** |
| Hành động | 2 chiến dịch | 3 chiến lược | **4 chiến lược** |

* **Toán học:** K=2 (hoặc K=3) cho Silhouette tốt hơn.
* **Nghiệp vụ:** K=4 tách được 4 phân khúc hành động rõ ràng: *VIP Store/Catalog (C1)*, *Trung lưu săn deal trên Web (C2)*, *Tiết kiệm gắn bó ngắn (C0)* và *Tiết kiệm trung thành lướt Web (C3)*. Medoid là khách hàng thực → dễ trình bày stakeholder và tránh lãng phí ngân sách (ví dụ: không đẩy quảng cáo số cho C1, không tung deal generic cho C0/C3).

**Khuyến nghị:** Dùng kết quả **K = 4** (`customer_personality_clustered_v2.csv`) làm cơ sở triển khai Marketing Automation.
