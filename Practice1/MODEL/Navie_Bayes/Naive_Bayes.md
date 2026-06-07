# Phân loại Email Spam: Naive Bayes

Tài liệu này hướng dẫn chi tiết về mã nguồn xây dựng thuật toán **Multinomial Naive Bayes** từ đầu bằng Python, NumPy và Pandas. Mô hình được sử dụng cho bài toán phân loại email thành hai lớp: **Spam (1)** và **Ham (0)**.
---

## Mục lục

1. [Giới thiệu chung](#giới-thiệu-chung)
2. [Yêu cầu hệ thống & cài đặt](#yêu-cầu-hệ-thống--cài-đặt)
3. [Chuẩn bị dữ liệu](#chuẩn-bị-dữ-liệu)
4. [Cấu trúc mã nguồn](#cấu-trúc-mã-nguồn)
5. [Kết quả thực nghiệm](#kết-quả-thực-nghiệm)
6. [Kết quả cuối cùng](#kết-quả-cuối-cùng)
7. [Tổng kết](#tổng-kết)

---

## Giới thiệu chung

Đoạn mã được phát triển dưới dạng Jupyter Notebook, gồm các bước chính:

- Đọc dữ liệu email đã được vector hóa sẵn.
- Sử dụng dữ liệu đặc trưng TF-IDF với **10.000 đặc trưng**.
- Xây dựng mô hình **Multinomial Naive Bayes From Scratch**.
- Huấn luyện mô hình bằng cách tính:
  - Xác suất tiên nghiệm của từng lớp: `P(Ham)`, `P(Spam)`.
  - Xác suất có điều kiện của từng đặc trưng theo từng lớp: `P(feature | Ham)`, `P(feature | Spam)`.
- Dự đoán nhãn bằng cách chọn lớp có xác suất hậu nghiệm cao hơn.
- Đánh giá mô hình bằng các chỉ số: Accuracy, Precision, Recall, F1-score, Log Loss và Confusion Matrix.

---

## Yêu cầu hệ thống & Cài đặt

Môi trường thực thi được áp dụng là Python 3.10+ và các thư viện cơ bản.

Cài đặt nhanh thông qua `pip`:

Các thư viện chính được sử dụng:

- `numpy`: xử lý mảng số và tính toán xác suất.
- `pandas`: hiển thị bảng kết quả.
- `joblib`: đọc dữ liệu đã lưu dưới dạng `.pkl`.
- `pathlib`: kiểm tra đường dẫn file.

---

## Chuẩn bị dữ liệu

Dữ liệu sử dụng là bộ dữ liệu email đã cân bằng và đã được vector hóa sẵn.

Các file đầu vào:

- `X_train_balance.pkl`: đặc trưng train sau vector hóa.
- `X_test_balance.pkl`: đặc trưng test sau vector hóa.
- `y_train_balance.pkl`: nhãn train.
- `y_test_balance.pkl`: nhãn test.


## Cấu trúc mã nguồn

Mã nguồn được thiết kế thành 3 khối xử lý chính.

### 1. Hàm load dữ liệu

Chương trình sử dụng `joblib` để đọc các file `.pkl`:

Hàm `load_file()` có nhiệm vụ kiểm tra file có tồn tại hay không. Nếu không tìm thấy file, chương trình sẽ báo lỗi rõ ràng.

---

### 2. Class `MultinomialNaiveBayesFromScratch`

Đây là phần chính của mô hình. Class này gồm các hàm:

#### `__init__(alpha=1.0)`

Khởi tạo tham số cho mô hình.


#### `fit(X, y)`

Hàm huấn luyện mô hình.

Trong quá trình huấn luyện, mô hình tính:

- Prior probability: xác suất xuất hiện của từng lớp.
- Conditional probability: xác suất từng đặc trưng xuất hiện trong từng lớp.

Công thức smoothing:

```python
smoothed_count = feature_count + self.alpha
```

Việc cộng thêm `alpha` giúp tránh trường hợp xác suất bằng 0 khi một đặc trưng không xuất hiện trong một lớp.

---

#### `predict_log_proba(X)`

Hàm này tính xác suất log cho từng lớp.

Mô hình sử dụng log probability để tránh lỗi tràn số khi nhân nhiều xác suất nhỏ.

Sau đó dùng kỹ thuật `log-sum-exp` để chuẩn hóa xác suất.

---

#### `predict_proba(X)`

Hàm này chuyển log probability về xác suất thông thường:

```python
return np.exp(self.predict_log_proba(X))
```

Kết quả dùng để tính **Log Loss**.

---

#### `predict(X)`

Hàm dự đoán nhãn cuối cùng.

Mô hình chọn lớp có xác suất lớn nhất:

```python
return self.classes_[np.argmax(log_proba, axis=1)]
```

Nếu xác suất Spam lớn hơn Ham thì dự đoán là Spam, ngược lại dự đoán là Ham.

---

### 3. Hàm đánh giá mô hình

Chương trình tự xây dựng các hàm đánh giá thay vì phụ thuộc hoàn toàn vào thư viện có sẵn.

#### `confusion_values(y_true, y_pred)`

Hàm này tính bốn thành phần của Confusion Matrix:

- **TP**: Spam thật và dự đoán đúng là Spam.
- **TN**: Ham thật và dự đoán đúng là Ham.
- **FP**: Ham thật nhưng bị dự đoán nhầm là Spam.
- **FN**: Spam thật nhưng bị dự đoán nhầm là Ham.

---

#### `evaluate(y_true, y_pred)`

Hàm này tính các chỉ số:

- Accuracy
- Precision
- Recall
- F1-score
- TN, FP, FN, TP

Công thức chính:

```python
accuracy = (TP + TN) / total
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)
```

---

#### `log_loss_score(y_true, y_proba, classes)`

Hàm này dùng để đánh giá chất lượng xác suất dự đoán của mô hình.

Log Loss càng thấp thì xác suất dự đoán càng tốt.

Lưu ý: Naive Bayes không có loss theo epoch như SVM. Vì vậy, Log Loss ở đây chỉ dùng để đánh giá sau khi mô hình đã train xong.

---