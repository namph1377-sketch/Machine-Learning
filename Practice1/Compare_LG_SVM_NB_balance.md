# BÁO CÁO TOÀN DIỆN THỬ NGHIỆM VÀ SO SÁNH CÁC MÔ HÌNH PHÂN LOẠI EMAIL SPAM (FROM SCRATCH)

## 1. Giới thiệu bài toán
Bài toán đặt ra là **Phân loại nhị phân (Binary Classification)** nhằm mục đích tự động nhận diện và phân tách các thư rác độc hại (Spam - Nhãn 1) ra khỏi các thư thông thường (Ham - Nhãn 0). Trong thực tế, dữ liệu thu thập từ các hệ thống email thường rơi vào trạng thái lệch dữ liệu nghiêm trọng (Data Imbalance) do số lượng thư thường lớn hơn rất nhiều so với thư rác. 

Báo cáo này tập trung vào việc hiện thực hóa, đánh giá và đối sánh hiệu năng của ba thuật toán học máy kinh điển được xây dựng hoàn toàn từ gốc (**From Scratch**): **Multinomial Naive Bayes**, **Support Vector Machine (SVM)**, và **Logistic Regression**. Điểm đặc biệt của thực nghiệm này là mô hình được huấn luyện trên một tập dữ liệu thô bị mất cân bằng (Imbalanced) nhưng được kiểm thử trên tập dữ liệu hoàn toàn cân bằng (Balanced) để kiểm định chính xác năng lực tổng quát hóa và khả năng chống chịu nhiễu của từng thuật toán.

---

## 2. Mục lục triển khai
1. Giới thiệu bài toán
2. Yêu cầu đặc thù về dữ liệu và xử lý số hóa văn bản
3. Danh mục các thư viện hỗ trợ nền tảng
4. Thuật toán áp dụng và Lý thuyết huấn luyện cơ bản
5. Cấu trúc chi tiết mã nguồn các lớp mô hình từ gốc
6. Phương thức đánh giá hiệu năng và kiểm định chỉ số
7. Kết quả thực nghiệm và Vai trò của trực quan hóa (Visualization)

---

## 3. Yêu cầu về dữ liệu và xử lý số hóa văn bản
Dữ liệu văn bản đầu vào không thể đưa trực tiếp vào mô hình toán học mà cần qua quy trình tiền xử lý và trích xuất đặc trưng số hóa nghiêm ngặt.

### 3.1. Phân bổ kích thước và trạng thái cân bằng dữ liệu
Hệ thống phân tách dữ liệu rõ ràng thành hai tập Train/Test độc lập với cấu trúc như sau:
* **Tập huấn luyện (Train Set - Lệch dữ liệu):** Bao gồm tổng cộng **4,396 mẫu**. Trong đó có **1,093 mẫu Spam** và **3,303 mẫu Ham**. Tỷ lệ lệch xấp xỉ $1:3$, phản ánh đúng bản chất phân bổ thực tế của dòng luân chuyển email.
* **Tập kiểm thử (Test Set - Cân bằng):** Bao gồm tổng cộng **1,639 mẫu**. Trong đó có **813 mẫu Spam** và **826 mẫu Ham**. Tập Test cân bằng giúp các chỉ số đo lường như Accuracy không bị ảo tưởng bởi lớp đa số.

### 3.2. Cấu hình TF-IDF Vectorization
Văn bản thô được số hóa thành ma trận thưa qua kỹ thuật tính toán trọng số TF-IDF (`TfidfVectorizer`) với các ràng buộc tham số nhằm tối ưu hóa không gian đặc trưng:
* `max_features=10000`: Giới hạn từ vựng giữ lại 10.000 từ có tần suất và trọng số cao nhất để kiểm soát chiều dữ liệu.
* `sublinear_tf=True`: Sử dụng hàm logarit toán học $1 + \log(	ext{tf})$ thay cho tần suất thô $	ext{tf}$ nhằm giảm thiểu tầm ảnh hưởng tiêu cực của các từ xuất hiện lặp đi lặp lại quá nhiều lần trong một email rác.
* `token_pattern=r' \w{2,} '`: Loại bỏ tất cả các ký tự đơn lẻ đứng độc lập, chỉ giữ lại các từ có độ dài từ 2 ký tự trở lên.
* `ngram_range=(1, 2)`: Khai thác cả từ đơn (Unigram) và cụm hai từ liền kề (Bigram) để bắt trọn ngữ cảnh ngữ pháp cục bộ.
* `min_df=3`: Loại bỏ các từ quá hiếm ho xuất hiện ít hơn 3 văn bản.
* `max_df=0.95`: Loại bỏ các từ quá phổ biến xuất hiện trong trên 95% số lượng văn bản (như từ dừng, quán từ).

*Lưu ý:* Đối với mô hình Logistic Regression, tập từ vựng mở rộng trích xuất từ bundle lưu trữ sẵn (`spam_model.pkl`) đạt kích thước đặc trưng cao hơn là **26,258 chiều**.

---

## 4. Các thư viện hỗ trợ nền tảng
Hệ thống module so sánh được xây dựng dựa trên sự hỗ trợ của các thư viện Python:
* `numpy`: Thực hiện các phép toán vector hóa, nhân ma trận thưa, tính toán hàm kích hoạt sigmoid và quản lý mảng đa chiều hiệu năng cao.
* `pandas`: Hỗ trợ đọc dữ liệu thô từ file CSV và cấu trúc bảng tổng hợp số liệu kết quả.
* `matplotlib.pyplot`: Đóng vai trò cốt lõi trong việc vẽ biểu đồ cột so sánh trực quan và trực quan hóa ma trận nhầm lẫn đa trục.
* `joblib`: Phục vụ công tác tuần tự hóa (serialization) giúp nạp và lưu trữ nhanh các ma trận đặc trưng (.pkl) cùng cấu trúc trọng số của mô hình đã huấn luyện sẵn.
* `scipy.sparse.issparse`: Kiểm tra và tối ưu các phép toán tích vô hướng (`.dot()`) trực tiếp trên cấu trúc ma trận thưa Scipy mà không cần giải nén sang ma trận đặc, giúp tiết kiệm bộ nhớ RAM.

---

## 5. Thuật toán áp dụng và Lý thuyết huấn luyện cơ bản

### 5.1. Thuật toán Multinomial Naive Bayes
Dựa trên Định lý Bayes với giả định ngây thơ về sự độc lập hoàn toàn giữa các từ đặc trưng trong văn bản:
$$P(C_k|x) \propto P(C_k) \prod_{i=1}^{n} P(x_i|C_k)$$
Mô hình ước lượng xác suất tiên nghiệm (Prior) của lớp và xác suất có điều kiện (Likelihood) của từng từ dựa trên tần suất xuất hiện trong tập huấn luyện. Để tránh lỗi xác xuất bằng 0 khi gặp từ mới trong tập Test, kỹ thuật làm mịn Laplace (**Laplace Smoothing**) với hệ số $ lpha = 0.1$ được áp dụng. Việc tính toán được thực hiện trong không gian Log (`log_prior` và `log_prob`) để chuyển phép nhân xác suất thành phép cộng, tránh hiện tượng tràn số dưới (underflow).

### 5.2. Thuật toán Support Vector Machine (SVM)
Mô hình SVM tuyến tính tìm kiếm một siêu phẳng (Hyperplane) phân cách tối ưu có lề lớn nhất (Maximum Margin) giữa hai lớp dữ liệu. Để đối phó với nhiễu và dữ liệu không thể phân tách hoàn hảo, mô hình áp dụng kỹ thuật **Soft-margin SVM** tối ưu hóa hàm lỗi Hinge Loss kết hợp thành phần chuẩn hóa L2 (Ridge Regularization):

Quy trình huấn luyện sử dụng thuật toán **Hạ Gradient ngẫu nhiên (Stochastic Gradient Descent - SGD)** cập nhật trọng số động theo từng mẫu dữ liệu:
* Nếu mẫu phân loại đúng và nằm ngoài vùng lề biên ($y_i \cdot 	ext{decision} \ge 1$), trọng số chỉ cập nhật thu hẹp biên theo hệ số phạt Regularization: $w \leftarrow w -  lpha \cdot (2\lambda w)$.
* Nếu mẫu bị phân loại sai hoặc vi phạm vùng lề biên an toàn ($y_i \cdot 	ext{decision} < 1$), trọng số và bias được cập nhật để sửa sai lỗi phân loại: $w \leftarrow w -  lpha \cdot (2\lambda w - y_i x_i)$ và $b \leftarrow b +  lpha \cdot y_i$.

### 5.3. Thuật toán Logistic Regression
Mô hình phân loại tuyến tính sử dụng hàm kích hoạt phi tuyến **Sigmoid** để ánh xạ đầu ra từ không gian số thực $(-\infin, +\infin)$ về không gian xác suất nằm trong khoảng $(0, 1)$:
Hàm mất mát được tối ưu hóa là **Binary Cross-Entropy Loss** bổ sung thành phần phạt chuẩn hóa L2 để kiểm soát hiện tượng quá khớp (Overfitting):
Mô hình sử dụng phương pháp **Hạ Gradient (Gradient Descent)** đồng thời trên toàn bộ tập dữ liệu qua các chu kỳ (iterations) liên tục để cập nhật vector trọng số $w$ và bias $b$.

---

## 6. Cấu trúc chi tiết mã nguồn các lớp mô hình từ gốc

Mã nguồn định nghĩa chi tiết 3 lớp đối tượng tương ứng với 3 thuật toán, đảm bảo chuẩn tương thích giao diện hàm của Scikit-Learn (`fit` và `predict`).

### 6.1. Lớp `MultinomialNaiveBayesFromScratch`
* **Tham số khởi tạo (`__init__`):** `alpha` (mặc định bằng 1.0) - hệ số làm mịn Laplace chống lỗi xác suất bằng 0.
* **Hàm `fit(X, y)`:**
    * Xác định danh sách các lớp mục tiêu độc nhất thông qua `np.unique(y)`.
    * Tính toán xác suất tiên nghiệm dạng logarit: `self.class_log_prior_[i] = np.log(X_c.shape[0] / n_samples)`.
    * Tính tổng tần suất xuất hiện của từ trong từng lớp bằng phép gom cụm ma trận thưa `X_c.sum(axis=0)`.
    * Áp dụng hệ số làm mịn và tính xác suất logarit có điều kiện của từ: `np.log(smoothed_count / smoothed_count.sum())`.
* **Hàm `predict_log_proba(X)`:** Thực hiện nhân ma trận đặc trưng với ma trận trọng số xác suất điều kiện đã học qua toán tử thưa: `X.dot(self.feature_log_prob_.T) + self.class_log_prior_`. Áp dụng mẹo Log-Sum-Exp để chuẩn hóa xác suất an toàn.
* **Hàm `predict(X)`:** Trả về nhãn lớp có xác suất hậu nghiệm logarit lớn nhất thông qua hàm `np.argmax(..., axis=1)`.

### 6.2. Lớp `SVMFromScratch`
* **Tham số khởi tạo (`__init__`):** `learning_rate` (tốc độ học $ lpha$), `lambda_param` (hệ số phạt L2 $\lambda$), `n_iters` (số lượng epoch lặp huấn luyện).
* **Hàm `fit(X, y)`:**
    * Ánh xạ mảng nhãn nhị phân dữ liệu thực tế từ dạng $\{0, 1\}$ sang dạng hình học của SVM là $\{-1, 1\}$ bằng toán tử `np.where(y <= 0, -1, 1)`.
    * Khởi tạo vector trọng số `self.w = np.zeros(n_features)` và `self.b = 0`.
    * Vòng lặp kép duyệt qua từng epoch và từng dòng dữ liệu thưa `X[idx]`.
    * Tính giá trị hàm quyết định: `decision = np.asarray(x_i.dot(self.w)).ravel()[0] + self.b`.
    * Kiểm tra điều kiện biên để đưa ra quyết định cập nhật trọng số cục bộ tùy theo nhánh lỗi. Chuyển đổi dòng thưa sang mảng đặc `x_i.toarray().ravel()` chỉ khi cần tính toán phép trừ vector ở nhánh phân loại sai.
* **Hàm `predict(X)`:** Dự đoán nhãn dựa trên dấu của hàm quyết định tuyến tính: `np.where(np.sign(linear_output) > 0, 1, 0)`.

### 6.3. Lớp `LogisticRegressionFromScratch`
* **Tham số khởi tạo (`__init__`):** `learning_rate` (tốc độ cập nhật), `max_iter` (số bước lặp tối đa), `lambda_` (trọng số phạt regularization), `threshold` (ngưỡng phân loại xác xuất).
* **Hàm `_sigmoid(z)`:** Tính giá trị hàm sigmoid kết hợp cơ chế giới hạn giá trị `np.clip(z, -250, 250)` để triệt tiêu hoàn toàn lỗi tràn số toán học (overflow).
* **Hàm `fit(X, y)`:** Khởi tạo các mảng trọng số bằng 0. Chạy vòng lặp tính toán đầu ra dự đoán `y_pred`, đo lường lỗi qua `_compute_loss`, tính toán ma trận đạo hàm riêng (Gradients) `dw` và `db` trên toàn bộ tập dữ liệu, sau đó dịch chuyển trọng số ngược chiều gradient.
* **Hàm `predict(X, threshold)`:** Trả về nhãn 1 nếu giá trị xác suất vượt ngưỡng `threshold`, ngược lại trả về nhãn 0.

---

## 7. Phương thức đánh giá hiệu năng và kiểm định chỉ số

### 7.1. Hiện thực hóa hàm tiện ích tính toán toán học `compute_metrics`
Hệ thống không sử dụng thư viện có sẵn để tính toán hiệu năng mà định nghĩa hàm độc lập dựa trên việc bóc tách thủ công Ma trận nhầm lẫn (Confusion Matrix):
* $TP$ (True Positive): Thực tế là Spam, dự đoán đúng là Spam.
* $FP$ (False Positive): Thực tế là Ham, dự đoán sai là Spam (Báo động giả - *đây là lỗi nguy hiểm nhất vì làm mất thư thường của người dùng*).
* $TN$ (True Negative): Thực tế là Ham, dự đoán đúng là Ham.
* $FN$ (False Negative): Thực tế là Spam, dự đoán sai là Ham (Lọt lưới thư rác).

Các metrics được áp dụng: Accuracy, Precision, Recall, F1-Score, FPR (False Positive Rate)

### 7.2. Kiểm định siêu tham số của các mô hình
* **Mô hình Naive Bayes:** Được cấu hình với hệ số điều chỉnh mịn `alpha=0.1` để tăng cường sức mạnh dự đoán từ vựng.
* **Mô hình SVM:** Được cấu hình tối ưu với `learning_rate=0.05`, hệ số regularization cực nhỏ `lambda_param=0.0001` để nới rộng lề phân cách, huấn luyện tinh gọn trong `n_iters=10` epoch để tránh overfit dữ liệu lệch.
* **Mô hình Logistic Regression:** Sử dụng ngưỡng phân loại tùy biến thực nghiệm được tinh chỉnh lên mức `threshold=0.63` (thay vì ngưỡng 0.5 mặc định) nhằm siết chặt độ chuẩn xác Precision, giảm tối đa tỷ lệ FPR để bảo vệ các email thông thường không bị chặn nhầm.

---

## 8. Kết quả thực nghiệm và Vai trò của trực quan hóa

### 8.1. Bảng số liệu tổng hợp kết quả thực nghiệm
Dưới đây là bảng số liệu chi tiết thu được sau khi kiểm thử cả ba mô hình trên cùng một tập Test cân bằng:

| Chỉ số đánh giá hiệu năng | Multinomial Naive Bayes | Support Vector Machine (SVM) | Logistic Regression |
| :--- | :---: | :---: | :---: |
| **Accuracy (Độ chính xác tổng thể)** | 98.66% | **99.57%** | 99.21% |
| **Precision (Độ chuẩn xác bắt Spam)** | 99.38% | **99.39%** | 99.38% |
| **Recall (Tỷ lệ bắt sạch Spam)** | 97.91% | **99.75%** | 99.02% |
| **F1-Score (Độ cân bằng mô hình)** | 98.64% | **99.57%** | 99.20% |
| **False Positive Rate (Tỷ lệ chặn nhầm) ↓** | **0.61%** | **0.61%** | **0.61%** |

### 8.2. Vai trò và ý nghĩa của việc trực quan hóa sơ đồ

Mã nguồn tổ chức trực quan hóa thông qua hai sơ đồ xuất ra file cứng: `confusion_matrices.png` và `metrics_comparison.png`. Việc trực quan hóa này mang ý nghĩa kỹ thuật quan trọng:

1.  **Phân tích sâu cấu trúc ma trận nhầm lẫn (`plot_confusion_matrix`):**
    * Giúp người phát triển nhận diện rõ số lượng mẫu cụ thể bị phân loại sai. Nhìn vào sơ đồ ma trận nhầm lẫn, cả ba mô hình đều đạt tỷ lệ chặn nhầm email thường cực kỳ thấp và đồng nhất ($FP = 5$ mẫu, tương ứng tỷ lệ báo động giả 0.61%). Đây là đích đến tối thượng của các hệ thống bộ lọc email thương mại, bảo vệ người dùng khỏi việc bị thất lạc các thư thông thường quan trọng.
    * Sự vượt trội về mặt hình học của SVM thể hiện ở ô bắt trúng thư rác ($TP = 811$ mẫu), chỉ để lọt lưới duy nhất đúng $FN = 2$ mẫu thư rác trên tổng số 813 mẫu kiểm thử, minh chứng cho sức mạnh tối ưu hóa lề mềm SGD từ gốc.
2.  **Định hình cán cân đánh giá qua biểu đồ cột so sánh chỉ số:**
    * Biểu đồ cột cụm cho thấy sự đồng đều và tiệm cận mức hoàn hảo ($>98\%$) của cả ba mô hình trên một tập dữ liệu cân bằng. Đồ thị trực quan giúp khẳng định cấu trúc SVM từ gốc đứng đầu toàn diện trên mọi khía cạnh đo lường chỉ số với điểm số F1-Score đạt mức kỷ lục 99.57% và độ chính xác tổng thể 99.57%, xếp ngay sau là Logistic Regression (99.21%) và cuối cùng là Naive Bayes (98.66%).