# BÁO CÁO TRIỂN KHAI MÔ HÌNH PHÂN CỤM K-MEANS FROM SCRATCH (OOP)

---

## 1. Giới thiệu bài toán
Trong quản trị quan hệ khách hàng (CRM) và Marketing dựa trên dữ liệu (Data-driven Marketing), **Phân khúc khách hàng (Customer Segmentation)** là một bài toán kinh điển thuộc nhóm Học không giám sát (Unsupervised Learning). 

Bài toán đặt ra yêu cầu khai phá cấu trúc tiềm ẩn từ một tập dữ liệu khách hàng chứa các thuộc tính hành vi bao gồm: **Tuổi tác (Age)**, **Thu nhập hàng năm (Annual Income)** và **Điểm chi tiêu (Spending Score)**. Mục tiêu cốt lõi là tự động gom nhóm các khách hàng có sự tương đồng lớn về hành vi vào cùng một phân khúc riêng biệt, giúp doanh nghiệp tối ưu hóa chiến lược chăm sóc khách hàng và cá nhân hóa chiến dịch tiếp thị.

---

## 2. Mục lục triển khai
1. [Yêu cầu về Dữ liệu đầu vào & Thư viện hỗ trợ](#3-yêu-cầu-về-dữ-liệu-đầu-vào--thư-viện-hỗ-trợ)
2. [Cơ sở Lý thuyết của Thuật toán K-Means](#4-cơ-sở-lý-thuyết-của-thuật-toán-k-means)
3. [Cải tiến Khởi tạo: Cơ chế tối ưu của K-Means++](#5-cải-tiến-khởi-tạo-cơ-chế-tối-ưu-của-k-means)
4. [Cấu trúc Mã nguồn Chi tiết (Class & Function Architecture)](#6-cấu-trúc-mã-nguồn-chi-tiết-class--function-architecture)
5. [Phương pháp Kiểm định và Tối ưu hóa Tham số K](#7-phương-pháp-kiểm-định-và-tối-ưu-hóa-tham-số-k)
6. [Quy trình Đọc kết quả và Đặc tả Phân khúc Khách hàng](#8-quy-trình-đọc-kết-quả-và-đặc-tả-phân-khúc-khách-hàng)

---

## 3. Yêu cầu về Dữ liệu đầu vào & Thư viện hỗ trợ

### 3.1. Yêu cầu về dữ liệu đầu vào (`preprocessed_customers.csv`)
Do công đoạn tiền xử lý dữ liệu được đảm nhiệm bởi một đội ngũ chuyên trách riêng biệt, mô hình K-Means này đặt ra các giả định và tiêu chuẩn bắt buộc đối với file dữ liệu đầu vào như sau:
* **Định dạng số hóa hoàn toàn (Fully Numerical):** Tất cả các thuộc tính định tính (ví dụ: Giới tính) phải được mã hóa về dạng số (Label Encoding hoặc One-Hot Encoding).
* **Chuẩn hóa biên độ (Feature Scaling):** Đây là yêu cầu tối quan trọng. Thuật toán K-Means đo lường độ tương đồng dựa trên khoảng cách hình học. Nếu các trường dữ liệu như Thu nhập (hàng chục nghìn USD) giữ nguyên biên độ so với Tuổi (từ 18 đến 70), trường Thu nhập sẽ áp đảo hoàn toàn toán thức tính khoảng cách. Dữ liệu cần được xử lý qua StandardScaler (Z-score normalization) hoặc MinMaxScaler trước khi nạp vào mô hình.
* **Xử lý khuyết thiếu và nhiễu (Missing values & Outliers):** Tập dữ liệu đảm bảo không chứa giá trị rỗng (NaN) và các điểm dị biệt cực đoan đã được lọc bỏ để tránh làm lệch tâm cụm.

### 3.2. Thư viện hỗ trợ tính toán và hiển thị
Mô hình tuân thủ nghiêm ngặt yêu cầu không sử dụng các thư viện Machine Learning cao cấp (như scikit-learn) để can thiệp vào quá trình tính toán lõi. Các thư viện được dùng bao gồm:
* `numpy`: Thư viện nền tảng thực hiện các phép toán đại số tuyến tính, tính toán ma trận đa chiều bằng kỹ thuật Vectorization để tối ưu hóa hiệu năng tính toán.
* `pandas`: Sử dụng để đọc tệp tin CSV cấu trúc và hỗ trợ tổng hợp thông tin thống kê sau phân cụm.
* `matplotlib.pyplot` & `seaborn`: Phục vụ mục đích trực quan hóa các đồ thị hàm chi phí và biểu đồ phân phối.

---

## 4. Cơ sở Lý thuyết của Thuật toán K-Means

Thuật toán K-Means hoạt động theo cơ chế lặp (Iterative) nhằm tối thiểu hóa tổng bình phương khoảng cách từ mỗi điểm dữ liệu đến tâm cụm của nó (gọi là Inertia hoặc Within-Cluster Sum of Squares - WCSS).

Quy trình toán học bao gồm 4 bước lặp tuần hoàn:
1. **Khởi tạo (Initialization):** Xác định trước số lượng cụm K và chọn ra K điểm làm các tâm cụm ban đầu (Centroids).
2. **Gán cụm (Assignment):** Với mỗi điểm dữ liệu X_i, tính khoảng cách hình học Euclidean tới tất cả K tâm cụm C_j:
   `d(X_i, C_j) = sqrt( sum( (X_im - C_jm)^2 ) )`
   Điểm X_i sẽ được gán vào cụm của tâm C_j gần nó nhất.
3. **Cập nhật tâm cụm (Update):** Sau khi toàn bộ các điểm đã được gán cụm, tọa độ mới của mỗi tâm cụm C_j được tính toán lại bằng trung bình cộng tọa độ của tất cả các điểm thuộc về cụm đó:
   `C_j = (1 / |S_j|) * sum(X_i thuộc S_j)`
   Trong đó S_j là tập hợp các điểm thuộc cụm j.
4. **Kiểm tra hội tụ (Convergence check):** So sánh sự thay đổi vị trí của các tâm cụm mới so với vòng lặp trước. Nếu khoảng cách dịch chuyển nhỏ hơn một ngưỡng dung sai `tol` xác định trước, hoặc thuật toán đạt tới số vòng lặp tối đa (`max_iters`), quy trình huấn luyện sẽ dừng lại.

---

## 5. Cải tiến Khởi tạo: Cơ chế tối ưu của K-Means++

### 5.1. Nhược điểm của cơ chế khởi tạo ngẫu nhiên thuần túy (Naive Random Selection)
Trong phiên bản K-Means truyền thống, K tâm cụm ban đầu được chọn hoàn toàn ngẫu nhiên từ tập dữ liệu. Điều này dẫn tới một rủi ro toán học rất lớn: **Sự hội tụ nghiệm cục bộ (Local Minima)**. Nếu vô tình các tâm cụm ban đầu được chọn quá gần nhau hoặc nằm tập trung trong cùng một nhóm dữ liệu tự nhiên, thuật toán sẽ mất rất nhiều vòng lặp để dịch chuyển, hoặc tệ hơn là phân cụm sai lệch hoàn toàn thực tế cấu trúc dữ liệu.

### 5.2. Giải pháp tối ưu từ K-Means++
Để khắc phục triệt để điểm yếu này, cấu trúc code hiện tại đã tích hợp thuật toán khởi tạo thông minh **K-Means++**. Nguyên lý cốt lõi của K-Means++ là: *"Các tâm cụm ban đầu nên nằm càng xa nhau càng tốt trong không gian hình học"*.

Quy trình chọn tâm cụm của K-Means++ bao gồm:
1. Chọn ngẫu nhiên tâm cụm đầu tiên C_1 từ tập dữ liệu.
2. Với mỗi điểm dữ liệu X_i, tính khoảng cách D(X_i) là khoảng cách ngắn nhất từ nó đến tâm cụm gần nhất đã được chọn.
3. Chọn tâm cụm tiếp theo từ các điểm dữ liệu với một hàm xác suất phân phối tỉ lệ thuận với bình phương khoảng cách:
   `P(X_i) = D(X_i)^2 / sum(D(X_j)^2)`
   Một điểm dữ liệu nằm càng xa các tâm cụm hiện tại thì xác suất nó được chọn làm tâm cụm tiếp theo càng cao.
4. Lặp lại bước 2 và 3 cho đến khi chọn đủ K tâm cụm.

> **Kết quả:** K-Means++ giúp mô hình hội tụ nhanh hơn một cách đáng kể (giảm số vòng lặp `fit`) và đảm bảo chất lượng nghiệm phân cụm tiệm cận tối ưu toàn cục.

---

## 6. Cấu trúc Mã nguồn Chi tiết (Class & Function Architecture)

Mô hình được tổ chức theo kiến trúc lập trình hướng đối tượng (OOP) thông qua lớp `KMeansFromScratch` phối hợp cùng hàm độc lập `silhouette_score_from_scratch`.

### 6.1. Lớp `KMeansFromScratch`
Lớp này đóng gói toàn bộ hành vi huấn luyện và dự đoán của thuật toán.

#### Các thuộc tính khởi tạo (`__init__`)
* `n_clusters` (default=3): Số lượng cụm K cần phân tách.
* `max_iters` (default=100): Giới hạn số vòng lặp tối đa để tránh vòng lặp vô hạn nếu dữ liệu dao động liên tục.
* `tol` (default=1e-4): Ngưỡng hội tụ (Tolerance). Nếu độ dịch chuyển của tất cả tâm cụm nhỏ hơn số này, mô hình tự động dừng.
* `random_state` (default=42): Giá trị seed cố định nhằm đảm bảo tính tái lập kết quả (reproducibility) trong các lần thực nghiệm.
* `centroids`: Lưu trữ tọa độ của K tâm cụm sau khi học xong (mảng Numpy).
* `labels`: Mảng chứa nhãn cụm (từ 0 đến K-1) tương ứng của từng mẫu dữ liệu huấn luyện.
* `inertia_`: Tổng bình phương khoảng cách lỗi (WCSS) dùng cho phương pháp Elbow.

#### Các phương thức xử lý nội bộ (Private Methods)
* `_init_centroids_plusplus(self, X)`: Hiện thực hóa lý thuyết K-Means++. Sử dụng toán tử mảng của Numpy để tính ma trận khoảng cách tối thiểu từ mọi điểm đến danh sách các centroid hiện tại và lấy mẫu theo trọng số xác suất `probs = dists**2 / sum(dists**2)`.
* `_assign_clusters(self, X)`: 
  * **Giải thuật:** Sử dụng kỹ thuật mở rộng chiều dữ liệu tự động (`np.newaxis`) của Numpy để thực hiện tính toán khoảng cách Euclidean đồng thời giữa N dòng dữ liệu với K tâm cụm mà không cần dùng vòng lặp `for`.
  * **Code lõi:** `np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)` sinh ra ma trận kích thước (N, K). Hàm áp dụng `np.argmin(..., axis=1)` để tìm chỉ số cột có khoảng cách nhỏ nhất, chính là nhãn cụm được gán.
* `_update_centroids(self, X, labels)`: Duyệt qua từng cụm j, trích xuất các điểm thỏa mãn điều kiện `X[labels == j]` và tính trung bình theo cột toán học thông qua hàm `.mean(axis=0)`. Hàm tích hợp mệnh đề kiểm tra an toàn `if np.any(labels == j)` để giữ nguyên vị trí cũ của centroid trong trường hợp cụm đó bị rỗng (tránh lỗi chia cho 0).

#### Các phương thức giao tiếp bên ngoài (Public Methods)
* `fit(self, X)`: Đóng vai trò là hàm quản lý dòng chảy điều khiển (Control Flow). Hàm gọi phương thức khởi tạo K-Means++, chạy vòng lặp cập nhật liên tục, lưu trữ bản sao `old_centroids = self.centroids.copy()` để đối chiếu điều kiện dừng dựa trên toán thức kiểm tra hội tụ `np.all(np.abs(new_centroids - old_centroids) < self.tol)`. Cuối cùng, hàm tính toán giá trị `self.inertia_`.
* `predict(self, X)`: Tiếp nhận ma trận dữ liệu mới và gọi hàm `_assign_clusters(X)` để trả về nhãn cụm. Phương thức tích hợp cơ chế kiểm tra lỗi trạng thái mô hình nếu thuộc tính `self.centroids` chưa được khởi tạo.

### 6.2. Hàm đánh giá độc lập `silhouette_score_from_scratch(X, labels)`
Hàm tính toán chỉ số Silhouette để đánh giá độ chuẩn xác của việc phân cụm độc lập với Class.
* **Cơ chế tính toán nội bộ:** * Đối với mỗi điểm dữ liệu i thuộc cụm A, hàm sử dụng ma trận broadcasting để tính `a(i)` - khoảng cách trung bình từ i đến tất cả các điểm khác trong cùng cụm A.
  * Tiếp tục tính khoảng cách trung bình từ điểm i đó đến toàn bộ các điểm thuộc một cụm ngoại lai B (với B khác A). Hàm lặp qua các cụm ngoại lai để tìm giá trị trung bình nhỏ nhất, ký hiệu là `b(i)`.
  * Điểm Silhouette của một điểm được định nghĩa bằng công thức: `s(i) = (b(i) - a(i)) / max(a(i), b(i))`.
  * Hàm trả về giá trị trung bình của `s(i)` trên toàn tập dữ liệu bằng lệnh `np.mean(np.nan_to_num(s))`.

---

## 7. Phương pháp Kiểm định và Tối ưu hóa Tham số K

Để xác định số lượng phân khúc khách hàng hợp lý nhất, cấu trúc code thiết lập một vòng lặp quét danh sách tham số K (Hyperparameter Tuning) chạy từ 2 đến 9. Kết quả của quá trình quét này được tổng hợp lên đồ thị hai trục Y song song (Dual Axis Plot).

### 7.1. Phương pháp Khuỷu tay (Elbow Method với đường Inertia)
* **Mục đích trực quan:** Vẽ đồ thị thể hiện mối quan hệ giữa số lượng cụm K và chỉ số Inertia (WCSS). Khi K tăng, Inertia chắc chắn sẽ giảm dần do khoảng cách tới các tâm cụm thu hẹp lại.
* **Điểm tối ưu:** Điểm tối ưu được xác định tại vị trí đường biểu diễn đột ngột thay đổi độ dốc mạnh mẽ (tạo thành một hình "khuỷu tay"). Tại đó, nếu tiếp tục tăng K thì tốc độ giảm WCSS không còn đáng kể, đồng nghĩa với việc mô hình bắt đầu bị quá khớp (overfitting).

### 7.2. Phương pháp Hệ số Dáng điệu (Silhouette Score Method)
* **Mục đích trực quan:** Hệ số Silhouette phản ánh độ phân tách rõ ràng giữa các cụm. Giá trị nằm trong khoảng [-1, 1].
* **Điểm tối ưu:** Chúng ta tìm kiếm giá trị K mà tại đó Silhouette Score đạt giá trị **đỉnh (Local Maximum)**. Điều này chứng tỏ các cụm vừa đạt được sự gắn kết cao bên trong, vừa phân tách rõ rệt với các cụm lân cận.

> **Mục tiêu của việc Visualize đồ thị đôi:** Giúp nhà phân tích đưa ra quyết định giao thoa (trade-off) hoàn hảo nhất. Một giá trị K lý tưởng là một điểm mà tại đó đồ thị Elbow xuất hiện "khuỷu tay" rõ ràng và đồ thị Silhouette Score đạt giá trị cao vượt trội.

---

## 8. Quy trình Đọc kết quả và Đặc tả Phân khúc Khách hàng

Khi tìm được tham số K tối ưu (trong bài toán mẫu cấu hình mặc định là `optimal_k = 5`), hệ thống tiến hành bước phân tích chân dung khách hàng:

1. **Thống kê mật độ (`value_counts`):** Xác định dung lượng thị trường của từng phân khúc (Cụm nào chiếm đa số, cụm nào thuộc nhóm thiểu số chuyên biệt).
2. **Hợp nhất dữ liệu đặc trưng (`groupby('Cluster').mean()`):** Tính toán giá trị trung bình của các thuộc tính gốc sinh học và thuộc tính hành vi (Tuổi, Thu nhập, Chi tiêu) cho từng cụm. 

### 8.1. Phương án phân cụm với K = 2 (Góc nhìn Tổng quan)

Với cấu hình $K = 2$, tập dữ liệu được phân chia thành hai thực thể đối lập rõ rệt trên không gian PCA thông qua một ranh giới phân tách cơ học dọc theo trục Thành phần chính 1 (PC1).

#### Bảng chỉ số trung tâm cụm (K = 2)
| Chỉ số chính | Cluster 0 | Cluster 1 |
| :--- | :---: | :---: |
| **Income (Thu nhập)** | **+0.81** | -0.50 |
| **Children (Con cái)** | **-0.64** | +0.40 |
| **MntWines (Chi tiêu Rượu)** | **+0.90** | -0.56 |
| **MntMeatProducts (Chi tiêu Thịt)** | **+0.91** | -0.56 |
| **NumDealsPurchases (Mua qua Deal)** | -0.19 | **+0.12** |
| **NumWebVisitsMonth (Lượt ghé Web)** | -0.70 | **+0.43** |

#### Đặc tả chân dung phân khúc

* **Cluster 0: Phân khúc Thượng lưu & Chi tiêu mạnh (Affluent High-Spenders)**
    * **Nhân khẩu học:** Nhóm khách hàng có thu nhập vượt trội (+0.81), độ tuổi trung bình nhỉnh hơn mặt bằng chung (+0.14) và phần lớn không có con nhỏ hoặc có rất ít con (-0.64).
    * **Hành vi tiêu dùng:** Mức chi tiêu cực kỳ cao trên tất cả các danh mục sản phẩm, đặc biệt dẫn đầu là Thịt (+0.91), Rượu vang (+0.90) và Cá (+0.81).
    * **Kênh tương tác:** Ưu tiên mua sắm qua kênh Catalog (+0.94) và Store truyền thống (+0.86). Tần suất ghé thăm Website rất thấp (-0.70), chứng tỏ họ chỉ truy cập Web khi có chủ đích mua sắm rõ ràng thay vì lướt xem tin tức. Nhóm này phản hồi rất tích cực với Campaign 5 (+0.45) và Campaign 1 (+0.38).
* **Cluster 1: Phân khúc Gia đình Tiết kiệm (Budget-Conscious Families)**
    * **Nhân khẩu học:** Các hộ gia đình có đông con nhỏ (+0.40) với mức thu nhập dưới trung bình (-0.50).
    * **Hành vi tiêu dùng:** Cắt giảm chi tiêu tối đa, toàn bộ các chỉ số chi tiêu cho các mặt hàng đều âm sâu (dao động từ -0.39 đến -0.56).
    * **Kênh tương tác:** Ghé thăm website của cửa hàng rất thường xuyên (+0.43) nhưng tỷ lệ chuyển đổi đơn hàng thấp. Họ nhạy cảm về giá và có xu hướng chỉ phát sinh giao dịch khi có các chương trình khuyến mãi, săn deal (+0.12).

---

### 8.2. Phương án phân cụm với K = 4 (Góc nhìn Chi tiết & Cá nhân hóa)

Khi mở rộng cấu hình lên $K = 4$, nhóm khách hàng "Ít tiền" (Cluster 1 ở phương án $K=2$) được giữ nguyên cấu trúc vì tính đồng nhất cao, trong khi nhóm "Có tiền" (Cluster 0 ở phương án $K=2$) được phân rã sâu sắc thành 3 nhóm khách hàng phụ (Sub-segments) có cá tính và hành vi mua sắm hoàn toàn khác biệt.

#### Bảng chỉ số trung tâm cụm (K = 4)
| Chỉ số đặc trưng | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 |
| :--- | :---: | :---: | :---: | :---: |
| **Income (Thu nhập)** | +0.15 | -0.67 | +0.83 | **+1.18** |
| **Children (Con cái)** | +0.39 | +0.36 | -0.82 | -1.02 |
| **MntWines (Chi rượu)** | +0.45 | -0.78 | +0.55 | **+1.70** |
| **MntFruits (Chi trái cây)** | -0.21 | -0.53 | **+1.08** | +0.74 |
| **MntMeatProducts (Chi thịt)** | -0.18 | -0.64 | +1.07 | **+1.33** |
| **NumDealsPurchases (Săn Deal)** | **+0.95** | -0.20 | -0.40 | -0.66 |
| **NumWebPurchases (Mua qua Web)**| **+0.85** | -0.73 | +0.43 | +0.49 |
| **NumWebVisitsMonth (Ghé Web)** | +0.36 | **+0.43** | -0.96 | -0.95 |
| **AcceptedCmp5 (Chiến dịch 5)** | -0.27 | -0.28 | -0.28 | **+3.37** |
| **Response (Phản hồi tổng)** | -0.04 | -0.18 | +0.01 | **+1.22** |

#### Đặc tả chân dung phân khúc

* **Cluster 0: Gia đình Trung lưu Săn Deal (Web-Centric Mid-Income Bargainers)**
    * **Nhân khẩu học:** Khách hàng trung niên có tuổi (+0.36), thời gian gắn bó dài (+0.37), thu nhập khá (+0.15) và có con nhỏ (+0.39).
    * **Hành vi:** Tiêu dùng ở mức trung bình, nhưng là "Vua săn deal" thực thụ (+0.95). Họ sở hữu số lượng giao dịch qua Website cao nhất hệ thống (+0.85). Đây là nhóm khách hàng chuyển đổi tốt thông qua các chiến dịch ưu đãi trực tuyến.
* **Cluster 1: Nhóm Tiết kiệm Triệt để (Hardcore Budget Consumers)**
    * **Hành vi:** Tương tự như nhóm Cluster 1 ở phương án $K=2$ nhưng ở mức độ cực đoan hơn. Thu nhập thấp nhất hệ thống (-0.67), gần như không chi tiêu cho bất kỳ mặt hàng nào, thường xuyên lướt web (+0.43) nhưng thậm chí không phản hồi với cả các chương trình giảm giá thông thường (-0.20).
* **Cluster 2: Nhà giàu Truyền thống / Tín đồ Thực phẩm Sạch (Wealthy Store-Shopping Purists)**
    * **Nhân khẩu học:** Khách hàng thu nhập rất cao (+0.83), hoàn toàn không có con nhỏ (-0.82).
    * **Hành vi:** Chi tiêu cực mạnh cho các nhóm mặt hàng thực phẩm tươi sống có lợi cho sức khỏe như Trái cây (+1.08), Thịt (+1.07), Cá (+1.17) và Đồ ngọt (+1.01). 
    * **Kênh tương tác:** Có xu hướng bài trừ công nghệ, điểm ghé thăm Web thấp kỷ lục (-0.96). Họ chỉ tin tưởng mua sắm trực tiếp tại Store cửa hàng (+0.83) hoặc đặt hàng truyền thống qua Catalog (+1.01). Hoàn toàn thờ ơ với quảng cáo số.
* **Cluster 3: Siêu VIP / Tín đồ Rượu vang & Khuyến mãi Cao cấp (Elite Wine Connoisseurs & Ad-Responders)**
    * **Nhân khẩu học:** Phân khúc đỉnh chóp của doanh nghiệp với thu nhập cao nhất hệ thống (+1.18), không vướng bận con cái (-1.02).
    * **Hành vi:** Sẵn sàng vung tiền không tiếc tay cho Rượu vang (+1.70) và các loại Thịt cao cấp (+1.33).
    * **Điểm cốt lõi:** Điểm đặc trưng tạo nên sự khác biệt của nhóm này là **độ nhạy quảng cáo cực kỳ lớn**. Họ phản hồi áp đảo với tất cả các Campaign, nổi bật là Campaign 5 (+3.37), Campaign 1 (+1.52), Campaign 4 (+1.20) và có tỷ lệ chốt đơn tổng thể cao nhất (+1.22).

---

### 3. Đánh giá và Đề xuất Lựa chọn Phương án

* **Về mặt Toán học:** Phương án $K = 2$ cho chỉ số Silhouette Score cao hơn, thể hiện cấu trúc phân tách giữa 2 miền dữ liệu là lớn và rõ ràng nhất. Tuy nhiên, nó lại giữ mức sai số nội cụm (Inertia/WCSS) lớn, khiến dữ liệu bị gộp quá thô.
* **Về mặt Ứng dụng Kinh doanh:** **Phương án $K = 4$ hoàn toàn tối ưu và vượt trội.** Việc bóc tách nhóm khách hàng có tiền ra thành 3 nhóm: *Gia đình săn deal trên Web (C0)*, *Người giàu truyền thống mua tại quầy (C2)*, và *Siêu VIP chuộng Rượu vang/Quảng cáo (C3)* giúp phòng Marketing có thể xây dựng các kịch bản cá nhân hóa (Personalization) chính xác, tránh lãng phí ngân sách quảng cáo vào nhóm C2 hoặc tung sai kênh tiếp cận với nhóm C0 và C3.

**Khuyến nghị:** Sử dụng kết quả phân cụm **$K = 4$** làm cơ sở dữ liệu cuối cùng để triển khai các chiến dịch Marketing Automation tiếp theo.