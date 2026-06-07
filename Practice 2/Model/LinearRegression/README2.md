# Product Sales Prediction Using Linear Regression


## Giới thiệu

Dự án này cài đặt hoàn toàn **Linear Regression từ đầu (không dùng mô hình sklearn)**.

Mục tiêu:

* Hiểu rõ cơ chế hoạt động của Linear Regression
* Tự xây dựng thuật toán Gradient Descent
* Hiểu quá trình tối ưu hàm mất mát
* Xây pipeline ML hoàn chỉnh

---

## Tổng quan pipeline

```text
Dữ liệu thô
   ↓
Tiền xử lý (Preprocessing)
   ↓
Linear Regression (tự xây dựng)
   ↓
Gradient Descent
   ↓
Dự đoán doanh số
   ↓
Đánh giá mô hình
```

---

## Kiến trúc mô hình

```text
        X (96,000 x 79)
              ↓
     ŷ = X · w + b
              ↓
     Loss = MSE(ŷ, y)
              ↓
     Gradient Descent
     w ← w - α · ∂L/∂w
     b ← b - α · ∂L/∂b
              ↓
     Lặp 300 epochs đến hội tụ
```

---

## Logic Linear Regression

### Công thức dự đoán:

```text
ŷ = X · w + b
```

### Hàm mất mát (MSE):

```text
L = (1/n) * Σ(ŷ_i - y_i)²
```

### Cập nhật trọng số:

```text
w ← w - α * (2/n) * Xᵀ(ŷ - y)
b ← b - α * (2/n) * Σ(ŷ - y)
```

---

## Tính năng chính

### Linear Regression (tự cài đặt)

* Gradient Descent thuần túy bằng numpy
* Theo dõi loss theo từng epoch
* Khởi tạo trọng số bằng zeros
* Hỗ trợ learning rate và số epochs tuỳ chỉnh

---

### Feature Engineering

* Z-score normalization cho 8 continuous features
* Giữ nguyên 71 one-hot features (binary 0/1)
* Tính mean/std chỉ trên tập train (tránh data leakage)

---

### Đánh giá mô hình

* MSE, RMSE, MAE trên tập test
* R² trên cả train và test
* Visualizations: Loss curve, Actual vs Predicted, Residual plot

---

## Dataset

Bài toán dự đoán doanh số bán lẻ:

| Loại dữ liệu | Ví dụ                                        |
| ------------ | -------------------------------------------- |
| Giá trị số   | unit_price, discount_pct, qty_roll_mean_30d  |
| Thời gian    | transaction_year, month, day, dayofweek      |
| Danh mục     | product_name, category, region, sales_channel|
| Khách hàng   | customer_segment, gender, age_group          |

---

## Kết quả mô hình

```text
Train:
R²    0.6965

Test:
MSE   0.3049
RMSE  0.5522
MAE   0.4479
R²    0.6948
```

---

## Minh hoạ hội tụ

### Loss curve

```text
Epoch   0  |  MSE = 1.000  (khởi đầu)
Epoch  20  |  MSE = 0.335  (giảm nhanh)
Epoch 100  |  MSE = 0.316
Epoch 200  |  MSE = 0.307
Epoch 280  |  MSE = 0.304  (hội tụ)
```

---

## Output

Sau khi train:

```text
retail_test_20_with_predictions.csv
```

Bao gồm:

* Dữ liệu gốc
* Cột nhãn thực tế: `y_test`
* Cột dự đoán: `y_pred`

---

## Cách chạy

### 1. Cài thư viện

```bash
pip install numpy pandas matplotlib
```

---

### 2. Train model

```bash
jupyter notebook LinearR.ipynb
```

---

## Cấu trúc project

```text
.
├── LinearR.ipynb
├── retail_train_80.csv
├── retail_test_20.csv
├── retail_test_20_with_predictions.csv
└── README.md
```

---

## Kiến thức áp dụng

* Linear Regression
* Gradient Descent
* Z-score Normalization
* Data Leakage Prevention
* Bias–Variance Tradeoff
* Mean Squared Error / R² Score
