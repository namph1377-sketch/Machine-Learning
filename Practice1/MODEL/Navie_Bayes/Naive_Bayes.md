# Spam Email Detection using Multinomial Naive Bayes From Scratch

## 1. Giới thiệu

Dự án xây dựng mô hình phân loại email **Spam/Ham** bằng thuật toán **Multinomial Naive Bayes** tự cài đặt từ đầu bằng **NumPy**. Dữ liệu đầu vào đã được vector hóa sẵn, vì vậy phần huấn luyện mô hình sử dụng trực tiếp các ma trận đặc trưng `X_train_balance`, `X_test_balance` và nhãn `y_train_balance`, `y_test_balance`.

Mục tiêu:

* Load dữ liệu đã vector hóa sẵn từ file `.pkl`.
* Kiểm tra kích thước dữ liệu train/test.
* Kiểm tra phân phối nhãn Spam/Ham.
* Xây dựng lớp `MultinomialNaiveBayesFromScratch`.
* Huấn luyện mô hình bằng xác suất tiên nghiệm và xác suất có điều kiện.
* Dự đoán nhãn Spam/Ham cho tập test.
* Đánh giá mô hình bằng Accuracy, Precision, Recall, F1-score, Log Loss và Confusion Matrix.

## 2. Dữ liệu đầu vào

Dữ liệu đã được xử lý và vector hóa trước khi đưa vào mô hình. Mỗi email được biểu diễn dưới dạng vector đặc trưng số.

Ý nghĩa nhãn:

| Label | Ý nghĩa |
|---:|---|
| `0` | Ham, không phải spam |
| `1` | Spam |

## 3. Load dữ liệu

Dữ liệu được load bằng `joblib`:
Mục đích của bước này:

* Đọc dữ liệu đã được lưu sẵn.
* Chuyển nhãn `y` về kiểu số nguyên.
* Đảm bảo dữ liệu đầu vào đúng định dạng trước khi huấn luyện.

## 4. Thuật toán Multinomial Naive Bayes
Naive Bayes là thuật toán phân loại dựa trên xác suất Bayes. Trong bài toán này, mô hình học xác suất của từng lớp và xác suất xuất hiện của từng đặc trưng trong mỗi lớp.
Công thức Bayes:
P(class | x) = P(x | class) * P(class) / P(x)
Trong phân loại, ta chọn lớp có xác suất hậu nghiệm lớn nhất:
ŷ = argmax P(class | x)
Vì `P(x)` giống nhau cho mọi lớp, mô hình chỉ cần so sánh:
P(class) * P(x | class)

## 5. Các xác suất mô hình học

### Prior Probability

Prior probability là xác suất xuất hiện của mỗi lớp trong tập train:


P(Spam) = số email Spam / tổng số email
P(Ham)  = số email Ham  / tổng số email
Trong code:

```python
self.class_log_prior_[i] = np.log(X_c.shape[0] / n_samples)
```
Mô hình dùng log probability để tránh lỗi tràn số hoặc xác suất quá nhỏ khi nhân nhiều xác suất lại với nhau.

### 6.Conditional Probability

Conditional probability là xác suất của từng đặc trưng khi biết email thuộc một lớp cụ thể:

```text
P(feature | Spam)
P(feature | Ham)
```
Ý nghĩa:

* `feature_count`: tổng giá trị của từng đặc trưng trong một lớp.
* `alpha`: tham số smoothing.
* `feature_log_prob_`: xác suất có điều kiện sau khi lấy log.

## 7. Laplace Smoothing

Tham số `alpha` dùng để tránh trường hợp xác suất bằng 0.

Nếu một từ hoặc đặc trưng không xuất hiện trong lớp Spam hoặc Ham, xác suất của nó có thể bằng 0. Khi nhân xác suất, chỉ cần một giá trị bằng 0 thì toàn bộ xác suất của lớp đó sẽ bằng 0. Vì vậy, smoothing được thêm vào để mô hình ổn định hơn.

Công thức:

```text
P(feature | class) = (count(feature, class) + alpha) / tổng count sau smoothing
```

## 8. Class MultinomialNaiveBayesFromScratch

Mô hình được xây dựng theo hướng lập trình hướng đối tượng, gồm các hàm chính:

| Hàm | Chức năng |
|---|---|
| `__init__()` | Khởi tạo tham số `alpha` và các biến cần học |
| `fit(X, y)` | Huấn luyện mô hình bằng tập train |
| `predict_log_proba(X)` | Tính log xác suất dự đoán |
| `predict_proba(X)` | Chuyển log xác suất về xác suất thường |
| `predict(X)` | Dự đoán nhãn cuối cùng |

Mục đích:

* `predict()` dùng để lấy nhãn dự đoán cuối cùng.
* `predict_proba()` dùng để lấy xác suất dự đoán của từng lớp.

## 9. Hàm đánh giá mô hình

Các giá trị trong Confusion Matrix:

| Ký hiệu | Ý nghĩa |
|---|---|
| TN | Dự đoán Ham và thực tế là Ham |
| FP | Dự đoán Spam nhưng thực tế là Ham |
| FN | Dự đoán Ham nhưng thực tế là Spam |
| TP | Dự đoán Spam và thực tế là Spam |

Các chỉ số đánh giá:

### Accuracy

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Đo tỷ lệ dự đoán đúng trên toàn bộ dữ liệu.

### Precision

```text
Precision = TP / (TP + FP)
```

Trong các email mô hình dự đoán là Spam, Precision cho biết có bao nhiêu email thực sự là Spam.

### Recall

```text
Recall = TP / (TP + FN)
```

Trong các email Spam thật, Recall cho biết mô hình phát hiện đúng được bao nhiêu email Spam.

### F1-score

```text
F1-score = 2 * Precision * Recall / (Precision + Recall)
```

F1-score là trung bình điều hòa giữa Precision và Recall.

### Log Loss

Log Loss đo mức độ sai lệch giữa xác suất dự đoán và nhãn thật. Giá trị càng thấp thì mô hình dự đoán xác suất càng tốt.


## 10. Kết luận

Mô hình **Multinomial Naive Bayes From Scratch** hoạt động rất tốt trên bài toán phân loại email Spam/Ham. Với dữ liệu đã được vector hóa và cân bằng, mô hình đạt kết quả cao trên tập test.

Ưu điểm của mô hình:

* Dễ hiểu và dễ cài đặt.
* Tốc độ huấn luyện nhanh.
* Phù hợp với dữ liệu văn bản đã vector hóa.
* Hiệu quả tốt trong bài toán phân loại Spam/Ham.

Hạn chế:

* Giả định các đặc trưng độc lập với nhau, trong khi thực tế các từ trong văn bản có thể liên quan đến nhau.
* Kết quả phụ thuộc vào chất lượng vector hóa dữ liệu trước đó.
* Nếu dữ liệu mới khác nhiều so với dữ liệu train, mô hình có thể cần được huấn luyện lại.
