# King County House Price Prediction Using Linear Regression (Gradient Descent)

## Giới thiệu

Dự án nhằm xây dựng mô hình **Linear Regression** để dự đoán **giá nhà** (`price`) from Scratch cho các ngôi nhà ở quận King, Washington, Hoa Kỳ, dựa trên đặc trưng diện tích, phòng ngủ, vị trí, grade, v.v.

Mục tiêu:

* Hiểu rõ cơ chế Linear Regression khi tối ưu bằng **Batch Gradient Descent**
* Tự xây dựng class `LinearRegression` với `fit()` / `predict()` và vòng lặp epoch
* Hiểu vai trò **feature scaling** với GD (tránh tràn số / NaN)
* Xây pipeline ML hoàn chỉnh và đánh giá trên **thang giá gốc USD** (giải ngược `log1p`)

---

## Pipeline

| # | Phase | Nội dung |
|---|-------|----------|
| 1 | Import thư viện | `pandas`, `numpy`, `matplotlib`, `pathlib` |
| 2 | Tải dữ liệu | Đọc `X_train/val/test.csv`, `y_train/val/test.csv`; drop `price_log`; tạo `y_train_log = log1p(price)` |
| 3 | Định nghĩa mô hình & metric | Class LR (Batch GD) + MSE, RMSE, MAE, MAPE, R² |
| 4 | Huấn luyện | `fit(X_train, y_train_log)` — cập nhật `w`, `b` theo epoch |
| 5 | Dự đoán & inverse | `predict` → `expm1` → giá gốc USD trên Train / Val / Test |
| 6 | Đánh giá & trực quan | Metric + Loss curve, Actual vs Predicted, Residual |

Notebook tương ứng: **`Model/LinearRegression_GD.ipynb`**.

---

## Kiến trúc mô hình

```text
        X_train (15,128 × 22)  — đã StandardScaler (Pre_housing)
              ↓
     Không scale lại (mean ≈ 0, std ≈ 1 trên train)
     Drop price_log khỏi X (tránh leakage)
              ↓
     Khởi tạo w = 0, b = 0
              ↓
     Lặp epochs lần (Batch GD):
         ŷ = X · w + b
         error = ŷ − y          # y = price_log
         w ← w − η · (2/n) · Xᵀ error
         b ← b − η · (2/n) · Σ error
         loss = mean(error²)   # MSE trên thang log
              ↓
     predict(X_*) → ŷ_log
              ↓
     ŷ_price = expm1(ŷ_log)    # e^ŷ_log − 1
              ↓
     Metric / plot trên giá gốc USD
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

Target lúc train: `price_log = log1p(price)` → loss curve phản ánh MSE trên **thang log**.

### Gradient (Batch GD)

```text
∂L/∂w = (2/n) · Xᵀ (ŷ − y)
∂L/∂b = (2/n) · Σ(ŷ − y)

w ← w − η · ∂L/∂w
b ← b − η · ∂L/∂b
```

### So với OLS

| | OLS (`LinearRegression.ipynb`) | GD (`LinearRegression_GD.ipynb`) |
|--|--|--|
| Cách tìm `w`, `b` | Normal Equation / `lstsq` (một lần) | Lặp nhiều epoch |
| Cần scale mạnh? | Ít nhạy hơn | **Rất quan trọng** (lr + feature lớn → overflow) |
| Hội tụ | Nghiệm tối ưu toàn cục (nếu ổn định số) | Phụ thuộc η, epochs, scale |
| Bias | Bias trick `X̃ = [1 \| X]` hoặc `b` riêng | `b` cập nhật riêng |
| Kiểm tra ma trận | Rank / condition number | Không cần (không invert) |

---

## Feature scaling (bắt buộc với GD)

Trong project này, **X đã được StandardScaler** trong `Scripts/Pre_housing.ipynb` (fit trên Train, transform Val/Test). Notebook GD:

* **Không scale lại** X
* Chỉ train trên `y_train_log = log1p(price)`
* Drop cột `price_log` khỏi X (nếu còn trong CSV) để tránh data leakage

| Vấn đề nếu bỏ sót scale | Hệ quả |
|-------------------------|--------|
| Feature thô (sqft, lat, zipcode_freq, …) chưa chuẩn hóa | Gradient cực lớn → MSE → `inf` → **NaN** |
| Learning rate quá lớn + feature chưa scale | Weights tràn số |

> **Lưu ý:** X_train sau load: `mean ≈ 0`, `std ≈ 1`. Missing values trên Train / Val / Test = **0**.

---

## Hyperparameters

| Tham số | Giá trị (trong notebook) | Ghi chú |
|---------|--------------------------|---------|
| Learning rate $\eta$ | `0.15` | Ổn định khi X đã StandardScaler |
| Epochs | `300` | Loss giảm nhanh rồi ổn định trên thang log |
| Khởi tạo | `w = 0`, `b = 0` | Phù hợp khi feature đã chuẩn hóa |
| Loss khi train | MSE trên `price_log` | In mỗi 20 epoch (`verbose_every=20`) |
| Final MSE (log) | ≈ `0.0567` | Sau ~300 epoch |
| Bias sau train | ≈ `13.047` | Gần mean của `price_log` |

Chỉ **`fit` một lần** trên train. Phase đánh giá chỉ `predict` — **không fit lại**.

---

## Dataset

Nguồn gốc: `Data/kc_house_data_NaN.csv` → tiền xử lý bởi `Scripts/Pre_housing.ipynb` (split **70 : 15 : 15**).

| | Train | Validation | Test |
|-|-------|------------|------|
| File X | `Data/X_train.csv` | `Data/X_val.csv` | `Data/X_test.csv` |
| File y | `Data/y_train.csv` | `Data/y_val.csv` | `Data/y_test.csv` |
| Số mẫu | 15,128 | 3,243 | 3,242 |
| Số features (sau drop `price_log`) | 22 | 22 | 22 |
| Missing values | 0 | 0 | 0 |

| Thành phần | Nội dung |
|------------|----------|
| `X_*.csv` | Feature đã **StandardScaler** (mean ≈ 0, std ≈ 1 trên Train); có thể còn cột `price_log` → notebook **drop** khi load |
| `y_*.csv` | Cột `price` — **giá gốc USD**, chưa log, chưa scale |
| Target train | `y_train_log = log1p(price)` |
| Target đánh giá | `y_*_price` (USD) sau `expm1` |

**22 feature dùng train:**  
`bedrooms`, `bathrooms`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `grade`, `sqft_above`, `sqft_basement`, `lat`, `long`, `sqft_living15`, `sqft_lot15`, `sqft_living_log`, `month_sold`, `age`, `is_renovated`, `years_since_renovated`, `has_basement`, `distance_to_center`, `zipcode_freq`

**Phạm vi giá (train):** khoảng `[75,000 , 7,700,000]` USD.

### Cấu hình đường dẫn

Notebook tự resolve thư mục `Data/` khi chạy từ `Model/` hoặc `Scripts/`:

```python
ROOT = Path.cwd()
if ROOT.name in ("Model", "Scripts"):
    ROOT = ROOT.parent
DATA_DIR = ROOT / "Data"
```

---

## Đánh giá (giải ngược log)

Model học trên thang log. Khi báo cáo nghiệp vụ:

```text
ŷ_log = model.predict(X)
ŷ     = expm1(ŷ_log)     # e^ŷ_log − 1
y     = price gốc USD
→ MSE / RMSE / MAE / MAPE / R² trên (y, ŷ) giá gốc
```

### Kết quả mô hình (giá gốc USD, sau `expm1`)

| Metric | Train | Val | Test | Diễn giải (Test) |
|--------|------:|----:|-----:|------------------|
| MSE | 26,897,055,895 | 27,792,855,625 | 35,252,305,431 | Sai số bình phương trung bình (USD²) |
| RMSE | 164,003 | 166,712 | 187,756 | Sai số điển hình ~\$187,756 |
| MAE | 100,335 | 100,623 | 106,002 | Sai số tuyệt đối trung bình ~\$106,002 |
| MAPE | 18.87% | 18.84% | 18.85% | Sai số % trung bình ~18.9% |
| R² | 0.7957 | 0.7884 | 0.7690 | Model giải thích ~76.9% phương sai trên test |

Train R² ≈ Val R² ≈ Test R² → không có dấu hiệu overfitting rõ.

> **Ghi chú:** Tham chiếu OLS (cùng data, cùng pipeline log/expm1): Test R² ≈ 0.77, MAPE ≈ 18.8%, RMSE ≈ \$188,190. GD với $\eta=0.15$, 300 epoch đạt mức tương đương OLS trên thang giá gốc.

### Trực quan hoá

* **Training Loss Curve** — MSE theo epoch (thang `price_log`)
* **Actual vs Predicted** — giá gốc USD (Test)
* **Residual Plot** — residual theo predicted (Test, USD)

---

## Cách chạy

```bash
# Từ thư mục gốc project
pip install numpy pandas matplotlib
jupyter notebook Model/LinearRegression_GD.ipynb
```

Yêu cầu dữ liệu đã có sẵn trong `Data/` (sinh bởi `Scripts/Pre_housing.ipynb`).

Chạy **Restart Kernel → Run All** theo đúng thứ tự phase (load → định nghĩa model → fit → predict/`expm1` → metric).

---

## Cấu trúc project (phần GD)

```text
.
├── Model/
│   ├── LinearRegression_GD.ipynb
├── Scripts/
│   └── Pre_housing.ipynb           # Tiền xử lý & split 70/15/15
├── Doc/
│   └── README.md           # Tài liệu notebook
│   └── Gen_Linear.md           # Tài liệu generate notebook
└── Data/
    ├── X_train.csv / y_train.csv
    ├── X_val.csv   / y_val.csv
    ├── X_test.csv  / y_test.csv
    └── kc_house_data_NaN.csv
```
