# Decision Tree Regressor From Scratch

## 1. Giới thiệu

Dự án này triển khai thuật toán **Decision Tree Regression** từ đầu bằng Python sử dụng NumPy và Pandas, không sử dụng các thư viện Machine Learning như Scikit-Learn.

Mô hình được xây dựng theo nguyên lý:

- Tìm điểm chia (split) tốt nhất dựa trên Variance Reduction.
- Tạo cây theo phương pháp đệ quy.
- Giá trị tại nút lá là trung bình của các mẫu thuộc nút đó.
- Hỗ trợ dự đoán cho bài toán hồi quy (Regression).

---

## 2. Cấu trúc chương trình

### 2.1. Lớp `Node`

Đại diện cho một nút trong cây quyết định.

#### Thuộc tính

- `feature`: chỉ số đặc trưng dùng để chia dữ liệu.
- `threshold`: giá trị ngưỡng chia.
- `left`: nút con bên trái.
- `right`: nút con bên phải.
- `value`: giá trị dự đoán tại nút lá.

### 2.2. Lớp `DecisionTreeRegressor`

#### Khởi tạo

```python
DecisionTreeRegressor(
    max_depth=10,
    min_samples_split=2
)
```

#### Tham số

| Tham số | Ý nghĩa |
|----------|----------|
| max_depth | Độ sâu tối đa của cây |
| min_samples_split | Số lượng mẫu tối thiểu để tiếp tục chia |

---

## 3. Nguyên lý hoạt động

### 3.1. Huấn luyện (`fit`)

```python
model.fit(X_train, y_train)
```

Chuyển dữ liệu sang NumPy Array và bắt đầu xây dựng cây từ nút gốc.

### 3.2. Xây dựng cây (`_grow_tree`)

#### Điều kiện dừng

- Đạt độ sâu tối đa.
- Số mẫu nhỏ hơn `min_samples_split`.
- Phương sai của tập dữ liệu bằng 0.

#### Khi dừng

```python
leaf_value = np.mean(y)
```

Giá trị nút lá được tính bằng trung bình của biến mục tiêu.

### 3.3. Tìm điểm chia tốt nhất (`_best_split`)

#### Thuật toán

1. Duyệt qua từng thuộc tính.
2. Lấy tất cả giá trị unique.
3. Thử từng giá trị làm ngưỡng chia.
4. Tính Variance Reduction.

#### Công thức

```text
Variance Reduction
=
Current Variance
-
(Variance Left + Variance Right)
```

#### Chi tiết

```text
VR = Var(parent)
     - [n_left * Var(left)
     + n_right * Var(right)]
```

Điểm chia có Variance Reduction lớn nhất sẽ được chọn.

### 3.4. Dự đoán (`predict`)

```python
y_pred = model.predict(X_test)
```

Mỗi mẫu sẽ được duyệt từ nút gốc tới nút lá theo điều kiện:

```python
if x[feature] <= threshold:
    đi sang trái
else:
    đi sang phải
```

Giá trị tại nút lá là kết quả dự đoán.

---

## 4. Dữ liệu sử dụng

### 4.1. File Train

```text
retail_train_80.csv
```

### 4.2. File Test

```text
retail_test_20.csv
```

### 4.3. Biến mục tiêu

```python
target_col = "sales_amount_log"
```

---

## 5. Huấn luyện mô hình

### 5.1. Mô hình với `max_depth = 10`

```python
model = DecisionTreeRegressor(
    max_depth=10,
    min_samples_split=2
)
```

### 5.2. Mô hình với `max_depth = 20`

```python
model_depth20 = DecisionTreeRegressor(
    max_depth=20,
    min_samples_split=2
)
```

### 5.3. Mô hình với `max_depth = 30`

```python
model_depth30 = DecisionTreeRegressor(
    max_depth=30,
    min_samples_split=2
)
```

---

## 6. Đánh giá mô hình

Các chỉ số đánh giá được sử dụng:

### 6.1. MAE (Mean Absolute Error)

```text
MAE = mean(|y - y_pred|)
```

Đo sai số tuyệt đối trung bình.

### 6.2. MSE (Mean Squared Error)

```text
MSE = mean((y - y_pred)^2)
```

Phạt mạnh các sai số lớn.

### 6.3. RMSE (Root Mean Squared Error)

```text
RMSE = sqrt(MSE)
```

Đưa sai số về cùng đơn vị với dữ liệu gốc.

### 6.4. R² Score

```text
R² = 1 - (SS_res / SS_tot)
```

#### Trong đó

```text
SS_res = Σ(y - y_pred)²
SS_tot = Σ(y - mean(y))²
```

#### Ý nghĩa

| Giá trị | Đánh giá |
|----------|----------|
| 1.0 | Dự đoán hoàn hảo |
| > 0.8 | Rất tốt |
| 0.6 - 0.8 | Tốt |
| 0.4 - 0.6 | Trung bình |
| < 0.4 | Kém |