# Gen_Linear.md — Hướng dẫn Xây dựng Lại (Rebuild Guide) Pipeline Mô hình Linear Regression (Gradient Descent)

**Mục tiêu của file này:**  
Khi bạn có một prompt mới (hoặc cần tái tạo ở thư mục/môi trường khác), **chỉ cần đọc file này** là có thể viết lại notebook gần như y hệt bản gốc (`Model/LinearRegression_GD.ipynb`) về logic, thứ tự phase, tên biến, class, hyperparameter, metric và biểu đồ.

File này **không** thay thế `Model/README.md` hay `Doc/LinearRegression.md` (các file đó giải thích ý nghĩa). File này tập trung vào **cách code chính xác** để rebuild.

**Notebook đích:** `Model/LinearRegression_GD.ipynb`  
**Tiền đề:** Đã chạy `Scripts/Pre_housing.ipynb` (hoặc theo `Doc/Gen_Pre_housing.md`) → có đủ CSV trong `Data/`.

---

## 0. Yêu cầu & Quy ước

### Thư viện bắt buộc
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
```

**Không** dùng `sklearn.linear_model` cho mô hình chính — tự implement Batch GD.  
**Không** cần `seaborn` / `train_test_split` / `StandardScaler` trong notebook model (đã xong ở preprocess).

### Quy ước biến (rất quan trọng — giữ nguyên tên)

| Biến | Ý nghĩa |
|------|---------|
| `ROOT`, `DATA_DIR` | Resolve path project / `Data/` |
| `DROP_COLS` | `["price_log"]` — loại khỏi X |
| `X_train_df`, `X_val_df`, `X_test_df` | DataFrame sau drop |
| `feature_names` | list tên 22 cột |
| `X_train`, `X_val`, `X_test` | `ndarray` float, shape `(n, 22)` |
| `y_train_price`, `y_val_price`, `y_test_price` | giá gốc USD (`ndarray`) |
| `y_train_log` | `log1p(y_train_price)` — **chỉ train** |
| `model` | instance `LinearRegression` (GD) |
| `train_pred_price_log`, `val_pred_price_log`, `test_pred_price_log` | dự đoán thang log |
| `train_pred_price`, `val_pred_price`, `test_pred_price` | `expm1(...)` → USD |
| `results_gd` | DataFrame metric Train/Val/Test |
| `residuals` | `y_test_price - test_pred_price` |

### Đường dẫn dữ liệu (tự động thích nghi)
```python
ROOT = Path.cwd()
if ROOT.name in ("Model", "Scripts"):
    ROOT = ROOT.parent
DATA_DIR = ROOT / "Data"
```

### Input bắt buộc (từ preprocess)
| File | Nội dung |
|------|----------|
| `Data/X_train.csv`, `X_val.csv`, `X_test.csv` | 23 cột đã StandardScaler (gồm `price_log`) |
| `Data/y_train.csv`, `y_val.csv`, `y_test.csv` | cột `price` — **giá gốc USD** |

### Output / kết quả mong đợi
- 22 feature sau drop `price_log`
- Shapes: Train **15128**, Val **3243**, Test **3242**
- X_train: `mean ≈ 0`, `std ≈ 1` (không scale lại)
- Missing = 0 trên mọi tập X
- Final MSE (log-space) ≈ **0.0567**
- Bias ≈ **13.047**
- Test (USD): R² ≈ **0.769**, MAPE ≈ **18.85%**, RMSE ≈ **187,756**

---

## 1. Cấu trúc notebook (thứ tự cell)

Rebuild đúng **5 phase** + markdown mục lục. Thứ tự cell gợi ý:

| # | Loại | Nội dung |
|---|------|----------|
| 0 | markdown | Title + bài toán + mục tiêu GD |
| 1 | markdown | Mục lục 5 phase |
| 2 | markdown | Phase 1 — mô tả data, bảng 70:15:15, lưu ý **không scale lại** |
| 3 | code | Import |
| 4 | code | Load 3 tập X/y, drop `price_log`, tạo `y_train_log` |
| 5 | markdown | Phase 2 — class GD + metrics |
| 6 | code | Class `LinearRegression` (Batch GD) |
| 7 | code | Hàm metric: `mse`, `rmse`, `mae`, `r2_score`, `mape`, `format_metric_table` |
| 8 | markdown | Phase 3 — pipeline 4 bước |
| 9 | markdown | 3.1 Train |
| 10 | code | `model = LinearRegression(lr=0.15, epochs=300)` + `fit` |
| 11 | markdown | 3.2 Predict log |
| 12 | code | `predict` 3 tập |
| 13 | markdown | 3.3 Inverse `expm1` |
| 14 | code | `expm1` → `*_pred_price` |
| 15 | markdown | Phase 4 — đánh giá USD |
| 16 | code | `results_gd` + `display` raw + formatted |
| 17 | markdown | Ghi chú train log / eval USD / so OLS |
| 18 | markdown | Phase 5 — bảng 3 biểu đồ |
| 19 | code | Training Loss Curve |
| 20 | code | Actual vs Predicted (Test) |
| 21 | code | Residual Plot (Test) |

---

## 2. Phase 1 — Chuẩn bị dữ liệu

### 2.1 Import
```python
# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
```

### 2.2 Load + tách target

**Quy tắc cứng:**
1. Drop `price_log` khỏi mọi `X_*.csv` (leakage / không dùng làm feature).
2. `y_*.csv` chỉ lấy cột `"price"` → giá gốc USD.
3. Target train: `y_train_log = np.log1p(y_train_price)` — **chỉ tạo cho train**.
4. Val/Test **không** cần `y_*_log` khi train; chỉ cần `y_*_price` để đánh giá.
5. **Không** StandardScaler lại; **không** fit model trên giá gốc.

```python
# LOAD DATASET — 3 tập train / validation / test
ROOT = Path.cwd()
if ROOT.name in ("Model", "Scripts"):
    ROOT = ROOT.parent
DATA_DIR = ROOT / "Data"

DROP_COLS = ["price_log"]  # X không chứa price_log (đó là target / leakage)

X_train_df = pd.read_csv(DATA_DIR / "X_train.csv").drop(columns=DROP_COLS)
X_val_df   = pd.read_csv(DATA_DIR / "X_val.csv").drop(columns=DROP_COLS)
X_test_df  = pd.read_csv(DATA_DIR / "X_test.csv").drop(columns=DROP_COLS)

feature_names = list(X_train_df.columns)

X_train = X_train_df.values.astype(float)
X_val   = X_val_df.values.astype(float)
X_test  = X_test_df.values.astype(float)

# Giá gốc (USD) — dùng cho đánh giá & biểu đồ
y_train_price = pd.read_csv(DATA_DIR / "y_train.csv")["price"].values.astype(float)
y_val_price   = pd.read_csv(DATA_DIR / "y_val.csv")["price"].values.astype(float)
y_test_price  = pd.read_csv(DATA_DIR / "y_test.csv")["price"].values.astype(float)

# Target huấn luyện: price_log = log1p(price)
y_train_log = np.log1p(y_train_price)

print(f"Số feature trong X: {len(feature_names)} (đã loại price_log)")
print(f"Features: {feature_names}")
print(f"X_train: {X_train.shape}, y_train_log: {y_train_log.shape}")
print(f"X_val:   {X_val.shape},   y_val_price:   {y_val_price.shape}")
print(f"X_test:  {X_test.shape},  y_test_price:  {y_test_price.shape}")
print(f"\nMissing — Train: {np.isnan(X_train).sum()}, Val: {np.isnan(X_val).sum()}, Test: {np.isnan(X_test).sum()}")
print(f"X_train mean≈{X_train.mean():.4f}, std≈{X_train.std():.4f} (đã StandardScaler)")
print(f"y_train_price range: [{y_train_price.min():,.0f}, {y_train_price.max():,.0f}] USD")
print(f"y_train_log   range: [{y_train_log.min():.4f}, {y_train_log.max():.4f}]")
```

### 2.3 Kiểm tra bắt buộc sau load

| Check | Giá trị mong đợi |
|-------|------------------|
| `len(feature_names)` | **22** |
| `X_train.shape` | `(15128, 22)` |
| `X_val.shape` | `(3243, 22)` |
| `X_test.shape` | `(3242, 22)` |
| Missing X (3 tập) | **0** |
| `X_train.mean()`, `.std()` | ≈ **0** và ≈ **1** |
| `y_train_price` range | ~`[75000, 7700000]` |
| `y_train_log` range | ~`[11.23, 15.86]` |

**22 feature (thứ tự sau drop `price_log`):**
```python
['bedrooms', 'bathrooms', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition',
 'grade', 'sqft_above', 'sqft_basement', 'lat', 'long', 'sqft_living15', 'sqft_lot15',
 'sqft_living_log', 'month_sold', 'age', 'is_renovated',
 'years_since_renovated', 'has_basement', 'distance_to_center', 'zipcode_freq']
```

> Lưu ý: Trong CSV gốc, `price_log` đứng giữa `sqft_lot15` và `sqft_living_log`. Sau `.drop(columns=["price_log"])` danh sách trên là đúng.

---

## 3. Phase 2 — Định nghĩa mô hình & metric

### 3.1 Class `LinearRegression` (Batch Gradient Descent)

**Yêu cầu implement:**
- `__init__(learning_rate=0.01, epochs=1000)` — default trong class; **khi train dùng 0.15 / 300**
- `fit(X, y, verbose_every=100)`: Batch GD full-batch mỗi epoch
- `predict(X)`: `X @ weights + bias`
- Lưu `self.weights`, `self.bias`, `self.loss_history`

**Công thức (bắt buộc đúng hệ số 2/n — MSE gradient):**
```text
ŷ = X · w + b
error = ŷ − y
dw = (2/n) · Xᵀ · error
db = (2/n) · Σ error
w ← w − η · dw
b ← b − η · db
loss = mean(error²)
```

```python
class LinearRegression:
    """Linear Regression optimized by batch Gradient Descent."""

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y, verbose_every=100):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for epoch in range(self.epochs):
            y_pred = X @ self.weights + self.bias
            error = y_pred - y

            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            loss = np.mean(error ** 2)
            self.loss_history.append(loss)

            if verbose_every and epoch % verbose_every == 0:
                print(f"Epoch {epoch:4d} | MSE (log-space) = {loss:.6f}")

        print(f"Training complete — final MSE (log) = {self.loss_history[-1]:.6f}")
        print(f"Bias = {self.bias:.6f}")
        print(f"Weights: shape={self.weights.shape}")

    def predict(self, X):
        return X @ self.weights + self.bias
```

**Không được:**
- Mini-batch / SGD (bản hiện tại là **full batch**)
- Bias trick `X_aug = [1|X]` (bias cập nhật **riêng**)
- Early stopping (chạy hết `epochs`)
- Regularization (L1/L2)

### 3.2 Metrics (from scratch)

Tính sau `expm1`, trên **giá USD**:

```python
# Evaluation Metrics

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def format_metric_table(df):
    """Format MSE/RMSE/MAE có dấu phẩy; R²/MAPE giữ 4 chữ số thập phân."""
    display_df = df.copy()
    for col in ["Train", "Val", "Test"]:
        formatted = []
        for metric, val in zip(display_df["Metric"], display_df[col]):
            if metric == "R²":
                formatted.append(f"{val:.4f}")
            elif metric == "MAPE":
                formatted.append(f"{val:.4f}%")
            else:
                formatted.append(f"{val:,.0f}")
        display_df[col] = formatted
    return display_df
```

**Thứ tự metric trong bảng:** `MSE`, `RMSE`, `MAE`, `MAPE`, `R²` (đúng thứ tự này).

---

## 4. Phase 3 — Huấn luyện & dự đoán

### Pipeline 4 bước (ghi trong markdown)
```text
X_scaled  →  fit(y_train_log)  →  pred_log  →  expm1()  →  pred_price (USD)
```

### 4.1 Hyperparameters cứng (khi instantiate)

| Tham số | Giá trị | Ghi chú |
|---------|---------|---------|
| `learning_rate` | **`0.15`** | Ổn định vì X đã scale |
| `epochs` | **`300`** | Loss ổn định sau ~100 epoch |
| `verbose_every` | **`20`** | In mỗi 20 epoch |
| Init `w`, `b` | **0** | Trong `fit()` |

```python
# Train trên price_log bằng Gradient Descent
# X đã StandardScaler → lr có thể lớn hơn khi feature chưa scale
model = LinearRegression(learning_rate=0.15, epochs=300)

print("Training...")
model.fit(X_train, y_train_log, verbose_every=20)
```

**Chỉ `fit` một lần trên train.** Không fit lại trên val/test.

### 4.2 Kỳ vọng log train (xấp xỉ)

| Epoch | MSE (log-space) |
|------:|----------------:|
| 0 | ~170.5 |
| 20 | ~0.058 |
| 100 | ~0.0567 |
| 280–299 | ~0.05670 |
| Final | **≈ 0.056700** |
| Bias | **≈ 13.047021** |
| Weights shape | `(22,)` |

### 4.3 Predict (thang log)
```python
# Dự đoán ra price_log
train_pred_price_log = model.predict(X_train)
val_pred_price_log   = model.predict(X_val)
test_pred_price_log  = model.predict(X_test)

print(f"Train pred_price_log: {train_pred_price_log.shape}")
print(f"Val pred_price_log:   {val_pred_price_log.shape}")
print(f"Test pred_price_log:  {test_pred_price_log.shape}")
```

### 4.4 Inverse log → USD

**Phải dùng `np.expm1`**, không dùng `np.exp` (vì target train là `log1p`).

```python
# Inverse: expm1(pred_log) → price gốc (USD)
train_pred_price = np.expm1(train_pred_price_log)
val_pred_price   = np.expm1(val_pred_price_log)
test_pred_price  = np.expm1(test_pred_price_log)
```

$$\text{price} = \mathrm{expm1}(\text{price\_log}) = e^{\text{price\_log}} - 1$$

---

## 5. Phase 4 — Đánh giá mô hình

Metric trên **giá gốc USD** cho cả 3 tập. Tên DataFrame: **`results_gd`**.

```python
# Metrics GD trên price gốc
results_gd = pd.DataFrame({
    "Metric": ["MSE", "RMSE", "MAE", "MAPE", "R²"],
    "Train": [
        mse(y_train_price, train_pred_price),
        rmse(y_train_price, train_pred_price),
        mae(y_train_price, train_pred_price),
        mape(y_train_price, train_pred_price),
        r2_score(y_train_price, train_pred_price),
    ],
    "Val": [
        mse(y_val_price, val_pred_price),
        rmse(y_val_price, val_pred_price),
        mae(y_val_price, val_pred_price),
        mape(y_val_price, val_pred_price),
        r2_score(y_val_price, val_pred_price),
    ],
    "Test": [
        mse(y_test_price, test_pred_price),
        rmse(y_test_price, test_pred_price),
        mae(y_test_price, test_pred_price),
        mape(y_test_price, test_pred_price),
        r2_score(y_test_price, test_pred_price),
    ],
})

print("=== Gradient Descent (raw) ===")
display(results_gd)

print("\n=== Gradient Descent (formatted) ===")
display(format_metric_table(results_gd))
```

### 5.1 Kết quả tham chiếu (bản hiện tại)

| Metric | Train | Val | Test |
|--------|------:|----:|-----:|
| MSE | 26,897,055,895 | 27,792,855,625 | 35,252,305,431 |
| RMSE | 164,003 | 166,712 | 187,756 |
| MAE | 100,335 | 100,623 | 106,002 |
| MAPE | 18.8729% | 18.8414% | 18.8520% |
| R² | 0.7957 | 0.7884 | 0.7690 |

**Dung sai chấp nhận khi rebuild:** R² Test trong khoảng **0.76–0.78**, MAPE ~**18.8%**, RMSE ~**1.87e5 – 1.89e5**. Nếu lệch lớn → kiểm tra drop `price_log`, `log1p`/`expm1`, lr/epochs, hoặc data CSV khác seed.

### 5.2 Ghi chú markdown (giữ nội dung)
- Train trên `price_log = log1p(price)` → giảm ảnh hưởng outlier.
- Đánh giá trên USD sau `expm1(pred_log)`.
- X đã StandardScaler → không inverse scale X; chỉ `expm1` cho y.
- Tham chiếu OLS (cùng data): Test R² ≈ 0.77, MAPE ≈ 18.8%, RMSE ≈ $188,190.

---

## 6. Phase 5 — Trực quan hoá

Đúng **3 biểu đồ**, thứ tự sau:

### 6.1 Training Loss Curve (log-space)
```python
# Training Loss Curve (log-space MSE)
plt.figure(figsize=(8, 4))
plt.plot(model.loss_history)
plt.title("Training Loss Curve — GD (MSE on price_log)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss (log-space)")
plt.grid(True)
plt.tight_layout()
plt.show()
```

### 6.2 Actual vs Predicted — Test (USD)
```python
# Actual vs Predicted — Test (price gốc)
plt.figure(figsize=(8, 8))
plt.scatter(y_test_price, test_pred_price, alpha=0.3)
plt.plot(
    [y_test_price.min(), y_test_price.max()],
    [y_test_price.min(), y_test_price.max()],
    "r--",
)
plt.xlabel("Actual Price (USD)")
plt.ylabel("Predicted Price (USD)")
plt.title("Actual vs Predicted — GD (Test)")
plt.grid(True)
plt.tight_layout()
plt.show()
```

### 6.3 Residual Plot — Test (USD)
```python
# Residual Plot — Test (price gốc)
residuals = y_test_price - test_pred_price

plt.figure(figsize=(8, 5))
plt.scatter(test_pred_price, residuals, alpha=0.3)
plt.axhline(y=0, color="r", linestyle="--")
plt.xlabel("Predicted Price (USD)")
plt.ylabel("Residual (USD)")
plt.title("Residual Plot — GD (Test)")
plt.grid(True)
plt.tight_layout()
plt.show()
```

**Title/label phải có `"GD"`** để phân biệt với notebook OLS.

---

## 7. Lưu ý quan trọng khi rebuild

1. **Không train trên `y_train_price`.** Chỉ train trên `y_train_log`.  
2. **Không dùng `np.exp` thay `np.expm1`.** Sai inverse → metric lệch mạnh.  
3. **Bắt buộc drop `price_log` khỏi X.** Nếu để lại → leakage (R² ảo cao).  
4. **Không scale lại X** trong notebook model (đã scale ở Pre_housing).  
5. **Gradient dùng `(2/n)`** (đạo hàm MSE). Đổi thành `(1/n)` vẫn hội tụ nhưng lr “tương đương” khác.  
6. **Full-batch GD** — mỗi epoch dùng toàn bộ `X_train`.  
7. **Chỉ fit 1 lần.** Val/Test chỉ `predict`.  
8. **Reproducibility data:** CSV phải từ preprocess `random_state=42` (xem `Gen_Pre_housing.md`). Đổi seed split → metric khác.  
9. Nếu MSE → `inf` / `NaN`: feature chưa scale hoặc `lr` quá lớn.  
10. Notebook chạy được từ `Model/` hoặc root nhờ block `ROOT` / `DATA_DIR`.

---

## 8. Checklist nhanh khi tái tạo

- [ ] Import: `pandas`, `numpy`, `matplotlib.pyplot`, `Path`
- [ ] Resolve `DATA_DIR` đúng khi cwd = `Model/` hoặc root
- [ ] Drop `price_log` → còn **22** feature
- [ ] Shapes 15128 / 3243 / 3242
- [ ] `y_train_log = log1p(y_train_price)`
- [ ] Class GD: init w=0, b=0; dw/db với `(2/n)`; `loss_history`
- [ ] Metric: MSE, RMSE, MAE, MAPE, R² + `format_metric_table`
- [ ] `learning_rate=0.15`, `epochs=300`, `verbose_every=20`
- [ ] Predict 3 tập → `expm1` → `results_gd`
- [ ] 3 plot: loss curve, actual vs pred, residual (Test, USD)
- [ ] Test R² ≈ 0.77, MAPE ≈ 18.85%, final log-MSE ≈ 0.0567

---

## 9. Skeleton tối giản (dùng cho prompt mới)

Copy khối dưới vào notebook/script mới và chạy (cần sẵn CSV trong `Data/`).

```python
# === GEN FROM Gen_Linear.md — LinearRegression_GD ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path.cwd()
if ROOT.name in ("Model", "Scripts"):
    ROOT = ROOT.parent
DATA_DIR = ROOT / "Data"

DROP_COLS = ["price_log"]

X_train_df = pd.read_csv(DATA_DIR / "X_train.csv").drop(columns=DROP_COLS)
X_val_df   = pd.read_csv(DATA_DIR / "X_val.csv").drop(columns=DROP_COLS)
X_test_df  = pd.read_csv(DATA_DIR / "X_test.csv").drop(columns=DROP_COLS)

feature_names = list(X_train_df.columns)
X_train = X_train_df.values.astype(float)
X_val   = X_val_df.values.astype(float)
X_test  = X_test_df.values.astype(float)

y_train_price = pd.read_csv(DATA_DIR / "y_train.csv")["price"].values.astype(float)
y_val_price   = pd.read_csv(DATA_DIR / "y_val.csv")["price"].values.astype(float)
y_test_price  = pd.read_csv(DATA_DIR / "y_test.csv")["price"].values.astype(float)
y_train_log   = np.log1p(y_train_price)

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y, verbose_every=100):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []
        for epoch in range(self.epochs):
            y_pred = X @ self.weights + self.bias
            error = y_pred - y
            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            loss = np.mean(error ** 2)
            self.loss_history.append(loss)
            if verbose_every and epoch % verbose_every == 0:
                print(f"Epoch {epoch:4d} | MSE (log-space) = {loss:.6f}")
        print(f"Training complete — final MSE (log) = {self.loss_history[-1]:.6f}")
        print(f"Bias = {self.bias:.6f}")

    def predict(self, X):
        return X @ self.weights + self.bias

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def format_metric_table(df):
    display_df = df.copy()
    for col in ["Train", "Val", "Test"]:
        formatted = []
        for metric, val in zip(display_df["Metric"], display_df[col]):
            if metric == "R²":
                formatted.append(f"{val:.4f}")
            elif metric == "MAPE":
                formatted.append(f"{val:.4f}%")
            else:
                formatted.append(f"{val:,.0f}")
        display_df[col] = formatted
    return display_df

model = LinearRegression(learning_rate=0.15, epochs=300)
model.fit(X_train, y_train_log, verbose_every=20)

train_pred_price = np.expm1(model.predict(X_train))
val_pred_price   = np.expm1(model.predict(X_val))
test_pred_price  = np.expm1(model.predict(X_test))

results_gd = pd.DataFrame({
    "Metric": ["MSE", "RMSE", "MAE", "MAPE", "R²"],
    "Train": [
        mse(y_train_price, train_pred_price),
        rmse(y_train_price, train_pred_price),
        mae(y_train_price, train_pred_price),
        mape(y_train_price, train_pred_price),
        r2_score(y_train_price, train_pred_price),
    ],
    "Val": [
        mse(y_val_price, val_pred_price),
        rmse(y_val_price, val_pred_price),
        mae(y_val_price, val_pred_price),
        mape(y_val_price, val_pred_price),
        r2_score(y_val_price, val_pred_price),
    ],
    "Test": [
        mse(y_test_price, test_pred_price),
        rmse(y_test_price, test_pred_price),
        mae(y_test_price, test_pred_price),
        mape(y_test_price, test_pred_price),
        r2_score(y_test_price, test_pred_price),
    ],
})
print(format_metric_table(results_gd))

plt.figure(figsize=(8, 4))
plt.plot(model.loss_history)
plt.title("Training Loss Curve — GD (MSE on price_log)")
plt.xlabel("Epoch"); plt.ylabel("MSE Loss (log-space)"); plt.grid(True); plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(y_test_price, test_pred_price, alpha=0.3)
plt.plot([y_test_price.min(), y_test_price.max()],
         [y_test_price.min(), y_test_price.max()], "r--")
plt.xlabel("Actual Price (USD)"); plt.ylabel("Predicted Price (USD)")
plt.title("Actual vs Predicted — GD (Test)"); plt.grid(True); plt.tight_layout(); plt.show()

residuals = y_test_price - test_pred_price
plt.figure(figsize=(8, 5))
plt.scatter(test_pred_price, residuals, alpha=0.3)
plt.axhline(y=0, color="r", linestyle="--")
plt.xlabel("Predicted Price (USD)"); plt.ylabel("Residual (USD)")
plt.title("Residual Plot — GD (Test)"); plt.grid(True); plt.tight_layout(); plt.show()

print("✅ GD model xong. Features:", len(feature_names), "| Test R²:",
      f"{r2_score(y_test_price, test_pred_price):.4f}")
```

---

## 10. Phụ lục — Rebuild OLS (`LinearRegression.ipynb`) nếu cần

Cùng Phase 1 load data + cùng metric/plot (bỏ loss curve). Khác:

| Mục | OLS |
|-----|-----|
| Class | `LinearRegressionOLS` |
| Fit | Bias trick `X_aug = [1 \| X]`, `theta = np.linalg.lstsq(X_aug, y, rcond=None)` |
| `bias` / `weights` | `theta[0]`, `theta[1:]` |
| Phase 2 | Thêm diagnostics `XTX`: det, rank, shape, cond trên Train |
| Hyperparams | Không có lr/epochs |
| Metric DF name | `results_ols` |
| Title plot | `"... — OLS (Test)"` |
| Test tham chiếu | R² ≈ 0.7679, MAPE ≈ 18.85%, RMSE ≈ 188,190 |

```python
class LinearRegressionOLS:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        ones = np.ones((X.shape[0], 1))
        X_aug = np.hstack([ones, X])
        theta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        self.bias = theta[0]
        self.weights = theta[1:]
        print(f"Bias = {self.bias:.6f}")
        print(f"Weights: shape={self.weights.shape}")

    def predict(self, X):
        return X @ self.weights + self.bias

# Diagnostics (Train)
ones = np.ones((X_train.shape[0], 1))
X_aug = np.hstack([ones, X_train])
XTX = X_aug.T @ X_aug
print("det", np.linalg.det(XTX), "rank", np.linalg.matrix_rank(XTX),
      "shape", XTX.shape, "cond", np.linalg.cond(XTX))

model = LinearRegressionOLS()
model.fit(X_train, y_train_log)
# sau đó predict → expm1 → metrics giống GD
```

---

## 11. Gợi ý prompt khi mở box mới

```text
Đọc Doc/Gen_Linear.md và xây lại Model/LinearRegression_GD.ipynb
y hệt theo guide: giữ nguyên tên biến, class LinearRegression (Batch GD),
hyperparameter lr=0.15 epochs=300, drop price_log, train trên log1p(price),
đánh giá sau expm1 trên 3 tập, metrics MSE/RMSE/MAE/MAPE/R²,
và 3 biểu đồ (loss curve, actual vs predicted, residual).
Tiền đề: Data/*.csv đã có từ Pre_housing (Gen_Pre_housing.md).
```

---

**Cách dùng file này hiệu quả nhất:**  
Mỗi lần prompt notebook mới (ví dụ `LinearRegression_GD_v2.ipynb`), đưa nguyên nội dung `Gen_Linear.md` + yêu cầu *"xây lại y hệt theo guide này, giữ nguyên tên biến, tham số, thứ tự phase"*.

Kết quả sẽ gần như giống bản hiện tại về logic, hyperparameter và metric (miễn data CSV không đổi).
