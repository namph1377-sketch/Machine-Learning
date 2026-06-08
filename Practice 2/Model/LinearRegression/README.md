# Product Sales Prediction Using Linear Regression


## Giới thiệu

Dự án này cài đặt hoàn toàn **Linear Regression từ đầu (không dùng thư viện có sẵn)**.

Mục tiêu:

* Hiểu rõ cơ chế hoạt động của Linear Regression
* Tự xây dựng thuật toán OLS (Ordinary Least Squares)
* Hiểu cách giải bài toán tối thiểu hóa MSE bằng Normal Equation
* Xây pipeline ML hoàn chỉnh

---

## Tổng quan pipeline

```text
Dữ liệu đã xử lý sẵn
   ↓
Load & Prepare Data
   ↓
Linear Regression (tự xây dựng — OLS)
   ↓
Normal Equation
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
     X̃ = [1 | X]   (thêm cột bias)
              ↓
     Normal Equation:
     θ = (X̃ᵀX̃)⁻¹ X̃ᵀy
              ↓
     ŷ = X · w + b
              ↓
     Nghiệm tối ưu toàn cục (một lần, không lặp)
```

---

## Logic Linear Regression

### Công thức dự đoán:

```text
ŷ = X · w + b
```

### Bias trick (gộp bias vào theta):

```text
X̃ = [1 | X]
ŷ = X̃ · θ      với θ = [b, w₁, w₂, ...]ᵀ
```

### Normal Equation (OLS):

```text
θ = (X̃ᵀ X̃)⁻¹ X̃ᵀ y
```

### Hàm mất mát (MSE):

```text
L = (1/n) * Σ(ŷ_i - y_i)²
```

---

## Tính năng chính

### Linear Regression (tự cài đặt — OLS)

* Normal Equation thuần túy bằng numpy
* Dùng `np.linalg.lstsq` để ổn định số học (tránh lỗi khi ma trận gần singular)
* Không cần learning rate, không cần epochs
* Nghiệm chính xác, tối ưu toàn cục

---

### Đánh giá mô hình

* MSE, RMSE, MAE trên tập test
* R² trên cả train và test
* Visualizations: Top 10 Feature Coefficients, Actual vs Predicted, Residual plot

---

## Dataset

Bài toán dự đoán doanh số bán lẻ:

| Loại dữ liệu | Ví dụ                                         |
| ------------ | --------------------------------------------- |
| Giá trị số   | unit_price, discount_pct, qty_roll_mean_30d   |
| Thời gian    | transaction_year, month, day, dayofweek       |
| Danh mục     | product_name, category, region, sales_channel |
| Khách hàng   | customer_segment, gender, age_group           |

---

## Kết quả mô hình

```text
Train:
R²    0.6987

Test:
MSE   0.3027
RMSE  0.5502
MAE   0.4495
R²    0.6971
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
jupyter notebook LinearRegression.ipynb
```

---

## Cấu trúc project

```text
.
├── LinearRegression.ipynb
├── LinearR.ipynb
├── retail_train_80.csv
├── retail_test_20.csv
├── retail_test_20_with_predictions.csv
└── README.md
```

---

## Kiến thức áp dụng

* Linear Regression
* Ordinary Least Squares (OLS)
* Normal Equation
* Bias Trick
* Bias–Variance Tradeoff
* Mean Squared Error / R² Score
