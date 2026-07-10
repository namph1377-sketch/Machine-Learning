# Product Sales Prediction Using Linear Regression (Gradient Descent)

## Giới thiệu

Dự án nhằm xây dựng mô hình **Linear Regression** để dự đoán **doanh số bán lẻ** (`sales_amount_log`) from Scratch của từng giao dịch dựa trên thông tin sản phẩm, khách hàng và giao dịch.

Mục tiêu:

* Hiểu rõ cơ chế Linear Regression khi tối ưu bằng **Batch Gradient Descent**
* Tự xây dựng class `LinearRegression` với `fit()` / `predict()` và vòng lặp epoch
* Hiểu vai trò **feature scaling** với GD (tránh tràn số / NaN)
* Xây pipeline ML hoàn chỉnh và đánh giá trên **thang doanh số gốc** (giải ngược log)

---

## Pipeline

| # | Phase | Nội dung |
|---|-------|----------|
| 1 | Import thư viện | `pandas`, `numpy`, `matplotlib` |
| 2 | Tải dữ liệu | Đọc `retail_train_80.csv` / `retail_test_20.csv`, xem shape & missing |
| 3 | Tiền xử lý | Tách X/y → kiểm tra scale → z-score continuous (giữ one-hot) |
| 4 | Định nghĩa mô hình & metric | Class LR (GD) + MSE, RMSE, MAE, R² |
| 5 | Huấn luyện | `fit(X_train, y_train)` — cập nhật `w`, `b` theo epoch |
| 6 | Đánh giá | `predict` → `np.exp` → metric & biểu đồ trên doanh số gốc |

Notebook tương ứng: **`Linear_Regression_(GD)`**.

---

## Kiến trúc mô hình

```text
        X_train (96,000 × 79)
              ↓
     Scale continuous (mean/std fit trên train)
     One-hot giữ nguyên 0/1
              ↓
     Khởi tạo w = 0, b = 0
              ↓
     Lặp epochs lần (Batch GD):
         ŷ = X · w + b
         error = ŷ − y
         w ← w − η · (2/n) · Xᵀ error
         b ← b − η · (2/n) · Σ error
         loss = mean(error²)   # MSE trên thang log
              ↓
     predict(X_test) → ŷ_log
              ↓
     ŷ_sales = exp(ŷ_log)     # giải ngược log
              ↓
     Metric / plot trên doanh số gốc
```

---

## Logic Linear Regression + Gradient Descent

### Công thức dự đoán

```text
ŷ = X · w + b
```

### Hàm mất mát (MSE)

```text
L = (1/n) · Σ(ŷᵢ − yᵢ)²
```

Target lúc train: `sales_amount_log` → loss curve phản ánh MSE trên **thang log**.

### Gradient (Batch GD)

```text
∂L/∂w = (2/n) · Xᵀ (ŷ − y)
∂L/∂b = (2/n) · Σ(ŷ − y)

w ← w − η · ∂L/∂w
b ← b − η · ∂L/∂b
```

### So với OLS (README)

| | OLS (`LinearRegression.ipynb`) | GD (`LR_GD.ipynb`) |
|--|--|--|
| Cách tìm `w`, `b` | Normal Equation / `lstsq` (một lần) | Lặp nhiều epoch |
| Cần scale mạnh? | Ít nhạy hơn | **Rất quan trọng** (lr + feature lớn → overflow) |
| Hội tụ | Nghiệm tối ưu toàn cục (nếu ổn định số) | Phụ thuộc η, epochs, scale |
| Bias | Bias trick `X̃ = [1 \| X]` | `b` cập nhật riêng |

---

## Feature scaling (bắt buộc với GD)

Trước khi train, notebook **chỉ scale 8 continuous features** (z-score theo **train**, áp dụng cho test):

* `unit_price`, `discount_pct`, `qty_roll_mean_30d`
* `transaction_year`, `transaction_month`, `transaction_day`, `transaction_dayofweek`
* `customer_age_group_encoded`

**71 one-hot / binary giữ nguyên 0/1.**

| Vấn đề nếu bỏ sót scale | Hệ quả |
|-------------------------|--------|
| `transaction_year` còn ~2024–2025 | Gradient cực lớn → MSE → `inf` → **NaN** |
| Learning rate quá lớn + feature chưa scale | Weights tràn số |

> Lưu ý triển khai: mỗi tên feature trong list `continuous_features` phải là **một chuỗi riêng** (có dấu phẩy). Thiếu `,` khiến Python dính chuỗi (vd. `qty_roll_mean_30dtransaction_year`) → cột không được scale.

---

## Hyperparameters

| Tham số | Giá trị (trong notebook) | Ghi chú |
|---------|--------------------------|---------|
| Learning rate $\eta$ | `0.25` | Ổn định **sau khi** scale continuous |
| Epochs | `200` | Loss giảm và ổn định trên thang log |
| Khởi tạo | `w = 0`, `b = 0` | Phù hợp khi feature đã chuẩn hóa |
| Loss khi train | MSE trên `sales_amount_log` | In mỗi 20 epoch |

Chỉ **`fit` một lần** trên train. Phase đánh giá chỉ `predict` — **không fit lại**.

---

## Dataset

| | Train | Test |
|-|-------|------|
| File | `retail_train_80.csv` | `retail_test_20.csv` |
| Số mẫu | 96,000 | 24,000 |
| Số features | 79 | 79 |
| Missing values | 0 | 0 |

| Loại dữ liệu | Số lượng | Ví dụ |
|---|---:|---|
| Continuous / numeric | 8 | `unit_price`, `discount_pct`, `qty_roll_mean_30d`, ngày/tháng/năm, nhóm tuổi |
| One-hot encoded | 71 | tên sản phẩm, danh mục, thương hiệu, giới tính, phân khúc KH, kênh bán, vùng, thanh toán |

**Target:** `sales_amount_log` — log-transform của doanh số thực.

Trong CSV, một số cột (`unit_price`, `qty_roll_mean_30d`) có thể đã z-score sẵn; các cột như `discount_pct`, `transaction_year` (2024–2025) **chưa scale** — notebook scale lại toàn bộ continuous trước GD.

---

## Đánh giá (giải ngược log)

Model học trên thang log. Khi báo cáo nghiệp vụ:

```text
ŷ_log = model.predict(X)
ŷ     = exp(ŷ_log)
y     = exp(y_log)
→ MSE / RMSE / MAE / R² trên (y, ŷ) doanh số gốc
```

### Kết quả mô hình (doanh số gốc, sau `exp`)

| Metric | Giá trị | Diễn giải |
|--------|---------|-----------|
| MSE | 81841.04 | Sai số bình phương trung bình trên thang sales gốc (test) |
| RMSE | 286.08 | Sai số điển hình ~286 đơn vị doanh số |
| MAE | 174.22 | Sai số tuyệt đối trung bình ~174 đơn vị doanh số |
| R² (Train) | 0.3476 | Model giải thích ~34.8% phương sai trên train (thang gốc) |
| R² (Test) | 0.3415 | Model giải thích ~34.2% phương sai trên test (thang gốc) |

Train R² ≈ Test R² → không có dấu hiệu overfitting rõ.

> **Ghi chú:** R² trên thang gốc thường **thấp hơn** R² trên thang log (OLS README ~0.70 log-scale) vì `exp` phóng đại sai số. So sánh công bằng với OLS nên cùng thang (log hoặc gốc).

### Trực quan hoá

* **Training Loss Curve** — MSE theo epoch (thang log)
* **Actual vs Predicted** — doanh số gốc
* **Residual Plot** — residual theo predicted (doanh số gốc)

---

## Cách chạy

```bash
pip install numpy pandas matplotlib
jupyter notebook LR_GD.ipynb
```

Chạy **Restart Kernel → Run All** theo đúng thứ tự phase (scale trước khi train).

---

## Cấu trúc project (phần GD)

```text
.
├── LR_GD.ipynb                 # Linear Regression + Gradient Descent
├── LinearRegression.md         # Tài liệu notebook GD (file này)
├── LinearRegression.ipynb      # Bản OLS (tham chiếu)
├── README.md                   # Tài liệu bản OLS
├── retail_train_80.csv
└── retail_test_20.csv
```

---

## Kiến thức áp dụng

* Linear Regression
* Batch Gradient Descent
* Learning rate & số epoch
* Feature scaling (z-score) cho continuous features
* One-hot encoding (giữ nguyên khi scale)
* Mean Squared Error / RMSE / MAE / R²
* Log-transform target và **inverse transform** (`np.exp`) khi đánh giá
* Phân biệt train (`fit`) vs inference (`predict`)
""")
