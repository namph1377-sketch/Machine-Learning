# BÁO CÁO PHÁT TRIỂN MÔ HÌNH PHÂN LOẠI EMAIL SPAM (SVM FROM SCRATCH)

## 1. Giới thiệu bài toán
Bài toán đặt ra là **Phân loại nhị phân (Binary Classification)** nhằm tự động nhận diện và lọc email rác (Spam) ra khỏi email thông thường (Ham). Việc xây dựng mô hình SVM từ gốc giúp kiểm soát hoàn toàn cơ chế tính toán ma trận thưa (Sparse Matrix), tối ưu hóa bộ nhớ khi làm việc với dữ liệu văn bản lớn, và hiểu sâu sắc cơ chế cập nhật trọng số bằng thuật toán hạ gradient ngẫu nhiên (SGD).

---

## 2. Mục lục triển khai
1. Giới thiệu bài toán
2. Yêu cầu về định dạng và xử lý dữ liệu
3. Thư viện hỗ trợ hệ thống
4. Lý thuyết toán học & Quy trình huấn luyện SVM
5. Cấu trúc chi tiết mã nguồn Mô hình
6. Phương thức kiểm định & Tối ưu hóa siêu tham số (Cross-Validation)
7. Đánh giá hiệu năng mô hình trên tập kiểm thử


---

## 3. Yêu cầu về định dạng dữ liệu
Để mô hình SVM có thể huấn luyện ổn định và đạt hiệu năng cao, dữ liệu đầu vào cần đáp ứng các tiêu chuẩn sau:
* **Định dạng ma trận**: Dữ liệu văn bản sau khi xử lý qua TF-IDF được lưu trữ dưới dạng **Ma trận thưa (Scipy Sparse Matrix)** để tiết kiệm dung lượng bộ nhớ. Kích thước tập Train là $6552 \times 10000$ và tập Test là $1639 \times 10000$ (với mật độ phần tử khác không cực kỳ tối ưu, chỉ chiếm **0.84%**).
* **Nhãn mục tiêu (Labels)**: Nhãn đầu vào ban đầu dạng nhị phân $0$ (Ham) và $1$ (Spam). Trước khi tính toán loss, nhãn sẽ được ánh xạ tự động sang tập $\{-1, 1\}$ để phù hợp với hàm mục tiêu hình học của SVM.
* **Phân bổ dữ liệu**: Tập dữ liệu đã được cân bằng hóa (Balanced Dataset) nhằm tránh hiện tượng mô hình bị thiên lệch:
    * **Tập Train**: 6,552 mẫu (3,249 Spam | 3,303 Ham).
    * **Tập Test**: 1,639 mẫu (813 Spam | 826 Ham).

---

## 4. Các thư viện hỗ trợ
Hệ thống tận dụng các thư viện bổ trợ tầng thấp và tầng cao bao gồm:
* `numpy`: Thực hiện các phép toán vector, tích vô hướng và quản lý mảng đa chiều.
* `scipy.sparse`: Xử lý chuyên biệt cho các hàng dữ liệu thưa (`issparse`, `vstack`) nhằm tránh crash bộ nhớ khi chuyển đổi ma trận đặc.
* `matplotlib.pyplot`: Trực quan hóa đường cong hàm mất mát (Loss curve).
* `pandas`: Hỗ trợ nạp dữ liệu thô từ cấu trúc file CSV.
* `sklearn.feature_extraction.text.TfidfVectorizer`: Chuyển đổi văn bản thô thành vector đặc trưng số theo trọng số TF-IDF kết hợp Unigram & Bigram.

---

## 5. Lý thuyết toán học & Quy trình huấn luyện SVM
Mô hình áp dụng thuật toán **Tối ưu hóa lề mềm (Soft-margin SVM)** sử dụng thuật toán **Hạ Gradient ngẫu nhiên (Stochastic Gradient Descent - SGD)**.

### Hàm mục tiêu (Objective Function)
Hàm mất mát tổng thể được tối ưu hóa trong mã nguồn có dạng:

$$J(w, b) = \frac{1}{n} \sum_{i=1}^{n} \max(0, 1 - y_i(w \cdot x_i + b)) + \lambda \|w\|^2$$

> **Lưu ý về Siêu tham số:** Hệ số Regularization $\lambda$ (`lambda_param`) có mối quan hệ nghịch đảo với tham số phạt $C$ trong các sách giáo khoa truyền thống ($C = \frac{1}{n \times \lambda}$). 
> * $\lambda$ càng lớn: Phạt nặng các trọng số có kích thước lớn, thu hẹp lề (margin).
> * $\lambda$ càng nhỏ: Nới rộng lề, chấp nhận lỗi huấn luyện nhiều hơn để tăng tính tổng quát hóa.

### Quy tắc cập nhật trọng số (SGD Update Rules)
Với mỗi mẫu dữ liệu $(x_i, y_i)$, mô hình tính giá trị hàm quyết định: $\text{decision} = w \cdot x_i + b$.
* **Trường hợp 1:** $y_i(w \cdot x_i + b) \ge 1$ (Mẫu nằm ngoài hoặc trên lề phân cách chính xác):
    $$\text{Hinge Loss} = 0$$
    Cập nhật trọng số chỉ dựa trên thành phần kiểm soát độ mượt (Regularization):
    $$w \leftarrow w - \alpha \cdot (2\lambda w)$$
* **Trường hợp 2:** $y_i(w \cdot x_i + b) < 1$ (Mẫu bị phân loại sai hoặc vi phạm vùng lề an toàn):
    $$\text{Hinge Loss} = 1 - y_i(w \cdot x_i + b)$$
    Cập nhật đồng thời cả Regularization và Hinge Loss để kéo biên phân cách về phía đúng:
    $$w \leftarrow w - \alpha \cdot (2\lambda w - y_i x_i)$$
    $$b \leftarrow b + \alpha \cdot y_i$$

*(Trong đó $\alpha$ là `learning_rate` - tốc độ học của mô hình).*

---

## 6. Cấu trúc chi tiết mã nguồn Mô hình
Lớp xử lý cốt lõi `SVMFromScratch` được cấu trúc bao gồm các phương thức chính sau:

### 6.1. Hàm khởi tạo `__init__`
* **Tham số đầu vào**: `learning_rate` (mặc định 0.001), `lambda_param` (mặc định 0.01), `n_iters` (mặc định 1000).
* **Biến nội tại**: Khởi tạo vector trọng số `w = None`, hệ số chặn `b = 0`, cùng hai mảng `epoch_log`, `loss_log` để lưu vết lịch sử huấn luyện nhằm phục vụ việc vẽ đồ thị.

### 6.2. Hàm huấn luyện `fit(X, y)`
* **Xử lý nhãn**: Chuyển đổi toàn bộ nhãn $0$ thành $-1$ bằng hàm `np.where`.
* **Khởi tạo**: Định hình kích thước mảng trọng số `self.w = np.zeros(n_features)` dạng Dense array.
* **Vòng lặp tối ưu hóa**: Duyệt qua từng epoch. Trong mỗi epoch, lặp qua từng dòng của ma trận thưa $X$. 
* **Phép toán thưa**: Sử dụng tích vô hướng `x_i.dot(self.w)` trực tiếp trên dòng thưa để tối ưu hiệu năng. Khi rơi vào trường hợp phân loại sai, dòng thưa `x_i` sẽ được giải nén cục bộ thành mảng đặc qua `x_i.toarray().ravel()` để thực hiện phép trừ vector.
* **Tính toán hàm mất mát cuối epoch**: Lưu giá trị loss trung bình kết hợp tổng bình phương trọng số vào lịch sử hệ thống.

### 6.3. Hàm dự đoán `predict(X)`
* **Cơ chế hoạt động**: Thực hiện tính toán song song trên toàn bộ ma trận dữ liệu kiểm thử `linear_output = X.dot(self.w) + self.b`.
* **Trả về kết quả**: Sử dụng hàm lấy dấu `np.sign`. Các mẫu có giá trị $> 0$ được gán nhãn $1$ (Spam), ngược lại gán nhãn $0$ (Ham).

---

## 7. Phương thức kiểm định & Tối ưu hóa tham số (Cross-Validation)

### Quy trình K-Fold Cross Validation tự cấu hình
Để tìm ra bộ tham số tối ưu nhất mà không làm rò rỉ dữ liệu (Data Leakage), mã nguồn xây dựng hàm `cross_validate_svm` với $K=5$ folds:
1. Chia đều $6552$ mẫu huấn luyện thành $5$ phần có kích thước tương đương.
2. Quét qua lưới tham số **Grid Search** gồm 27 tổ hợp từ các giá trị:
    * `learning_rate`: $[0.001, 0.01, 0.05]$
    * `lambda_param`: $[0.0001, 0.0005, 0.001]$
    * `n_iters`: $[10, 50, 70]$
3. Với mỗi tổ hợp, huấn luyện mô hình trên 4 phần và đánh giá độ chính xác (Accuracy) trên phần còn lại.

### Kết quả tìm kiếm siêu tham số tối ưu
Qua quá trình thực nghiệm, bộ tham số mang lại hiệu năng cao nhất trên tập kiểm định chéo bao gồm:
* **Learning Rate ($\alpha$)**: $0.05$
* **Lambda Param ($\lambda$)**: $0.0001$
* **Số lượng Epochs (`n_iters`)**: $70$
* **CV Accuracy đạt được**: **99.47%** (Độ lệch chuẩn cực thấp $\pm0.35\%$).

---

## 8. Đánh giá hiệu năng mô hình trên tập kiểm thử (Test Set)
Sau khi huấn luyện lại mô hình trên toàn bộ tập Train với bộ tham số tối ưu nhất, kiểm thử diện rộng trên tập Test thu được ma trận nhầm lẫn và các chỉ số hiệu năng vô cùng ấn tượng:

### Ma trận nhầm lẫn (Confusion Matrix)
| Thành phần | Ý nghĩa định danh bài toán | Số lượng mẫu thực tế |
| :--- | :--- | :--- |
| **True Positive (TP)** | Hệ thống bắt trúng chính xác email Spam | **812** |
| **False Positive (FP)** | Hệ thống chặn nhầm email thường thành Spam (Báo động giả) | **5** |
| **True Negative (TN)** | Hệ thống giữ lại chính xác email thường (Ham) | **821** |
| **False Negative (FN)** | Hệ thống bỏ sót (lọt lưới) email Spam | **1** |

### Các chỉ số đo lường hiệu năng
* **Accuracy (Độ chính xác tổng thể)**: **99.63%** $\rightarrow$ Mô hình phân loại chính xác hầu như tuyệt đối toàn bộ email.
* **Precision (Độ chuẩn xác bắt Spam)**: **99.39%** $\rightarrow$ Trong các email bị mô hình gắn nhãn Spam, tỷ lệ email rác thực sự chiếm đến 99.39%, hạn chế tối đa việc mất mát thông tin quan trọng của người dùng.
* **Recall (Độ nhạy / Tỷ lệ bắt sạch Spam)**: **99.88%** $\rightarrow$ Chỉ lọt lưới vỏn vẹn 2 email rác trên tổng số 813 email rác thực tế.
* **F1-Score**: **99.63%** $\rightarrow$ Giá trị trung bình điều hòa giữa Precision và Recall đạt trạng thái cân bằng lý tưởng.
* **False Positive Rate (Tỷ lệ chặn nhầm)**: **0.61%** $\rightarrow$ Đạt sát mục tiêu tối thượng của các hệ thống lọc mail ($< 1\%$).

