# Gen_LR — Hướng dẫn tái tạo Linear Regression (GD) giống phiên bản hiện tại

> **Mục đích file này:** Mỗi lần mở chat/prompt mới, chỉ cần đọc `Gen_LR.md` (+ tham chiếu data) là đủ để **xây lại notebook Linear Regression + Gradient Descent** tương thích phiên bản hiện tại (`Linear_Regression_(GD)`), không phụ thuộc transcript cũ.

| File | Vai trò |
|------|---------|
| `Gen_LR.md` | **Playbook xây dựng** (file này) — quy trình, quyết định, pitfalls, checklist |
| `LinearRegression.md` | Tài liệu sản phẩm / kiến thức mô hình (sau khi đã có notebook) |
| `Linear_Regression_(GD)` | Notebook chuẩn cần khớp |
| `retail_train_80.csv` / `retail_test_20.csv` | Data |

---

## 0. Prompt gợi ý cho chat mới

Copy vào box prompt khi cần agent build lại:

```text
Đọc Gen_LR.md và xây / sửa notebook Linear Regression + Batch Gradient Descent
from scratch (không sklearn.linear_model), khớp pipeline và quyết định trong Gen_LR.md.
Data: retail_train_80.csv, retail_test_20.csv. Target: sales_amount_log.
Output notebook: LR_GD.ipynb. Tài liệu mô tả: LinearRegression.md nếu cần cập nhật.
```

---

## 1. Bài toán & ràng buộc cứng

### Bài toán

* Dự đoán doanh số giao dịch bán lẻ.
* **Target train:** `sales_amount_log` (log của doanh số).
* **Đánh giá báo cáo:** trên **doanh số gốc** = `np.exp(...)`.

### Ràng buộc (không được phá)

1. **From scratch** — class tự viết `fit` / `predict`; **không** `sklearn.linear_model.LinearRegression`.
2. **Batch Gradient Descent** — không Normal Equation / `lstsq` (đó là bản OLS).
3. **Train/test đã tách sẵn** trong 2 file CSV — không tự `train_test_split` lại (trừ khi user yêu cầu).
4. **Scale:** chỉ continuous; **one-hot giữ 0/1**.
5. **Mean/std fit trên TRAIN**, transform cả train và test.
6. **`fit` chỉ 1 lần** lúc train; phase đánh giá chỉ `predict`.
7. Markdown đầu notebook: giới thiệu bài toán, mục tiêu, tổng quan data trước xử lý, mục lục.
8. Pipeline phase rõ ràng: Import → Load → Preprocess → Model+Metric → Train → Eval.

---

## 2. Dataset (trước xử lý trong notebook)

| | Train | Test |
|--|------:|-----:|
| File | `retail_train_80.csv` | `retail_test_20.csv` |
| Mẫu | 96,000 | 24,000 |
| Cột | 80 (79 features + target) | 80 |
| Missing | 0 | 0 |

* **Target:** `sales_amount_log` (~1.86 → 7.81 trên train).
* **8 continuous / numeric:**  
  `unit_price`, `discount_pct`, `qty_roll_mean_30d`,  
  `transaction_year`, `transaction_month`, `transaction_day`, `transaction_dayofweek`,  
  `customer_age_group_encoded`
* **~71 one-hot/binary:** gender, segment, product_name_*, category_*, brand_*, payment_*, channel_*, region_*

**Quan trọng:** Trong CSV, `unit_price` / `qty_roll_mean_30d` có thể đã z-score; `discount_pct`, `transaction_year` (2024–2025), tháng/ngày… **chưa scale**. GD **phải** scale continuous trước train.

---

## 3. Pipeline chuẩn (thứ tự cell)

Xây notebook theo đúng thứ tự này. Mỗi phase: **markdown mô tả** + **code**.

| # | Phase | Việc cần làm |
|---|--------|--------------|
| 0 | Intro markdown | Bài toán, mục tiêu notebook, data overview, mục lục |
| 1 | Import | `pandas`, `numpy`, `matplotlib.pyplot` |
| 2 | Load | `read_csv` train/test, print shape, missing, `head()` |
| 3.1 | Tách X/y | `target = "sales_amount_log"`; numpy float; test `reindex(columns=feature_names)` |
| 3.2 | Check scale | In min–max (và tên cột) từng feature **trước** scale |
| 3.3 | Feature scaling | Z-score 8 continuous; one-hot không đổi; in mean/std/range **sau** scale |
| 4.1 | Model class | `LinearRegression` GD + `loss_history` |
| 4.2 | Metrics | `mse`, `rmse`, `mae`, `r2_score` |
| 5 | Train | `model = LinearRegression(...); model.fit(X_train, y_train)` |
| 6.1 | Predict + inverse log | `predict` → `np.exp` cho y và ŷ (train & test) |
| 6.2 | Bảng metric | MSE/RMSE/MAE/R² **trên thang gốc** |
| 6.3 | Loss curve | `model.loss_history` (MSE **log**, lúc train) |
| 6.4 | Actual vs Predicted | scatter + đường y=x (thang gốc) |
| 6.5 | Residual plot | residual vs predicted (thang gốc) |

**Không** trộn OLS (determinant, rank, condition number, `lstsq`) vào notebook GD trừ khi user yêu cầu so sánh.

---

## 4. Quyết định thiết kế (đã chốt phiên bản hiện tại)

| Hạng mục | Quyết định | Lý do |
|----------|------------|--------|
| Thuật toán | Batch GD, MSE | Học iterative; loss = mean((ŷ−y)²) |
| Gradient | `dw = (2/n) X.T @ error`, `db = (2/n) sum(error)` | Đạo hàm MSE đầy đủ (hệ số 2) |
| Update | `w -= lr * dw`, `b -= lr * db` | Standard GD |
| Init | `w = 0`, `b = 0` | Ổn khi feature đã scale |
| Bias | Biến `b` riêng | Không bắt buộc bias trick |
| Scale | Manual mean/std, không bắt buộc StandardScaler | Kiểm soát one-hot không bị scale |
| Eval scale | `np.exp` sau predict | Metric nghiệp vụ trên sales thật |
| Train loss | Giữ trên log | Khớp target lúc `fit` |
| sklearn | Chỉ có thể import nếu cần util — **không** dùng LR của sklearn | From scratch |

### Hyperparameters hiện tại (code `LR_GD.ipynb`)

| Tham số | Giá trị |
|---------|--------:|
| `learning_rate` | `0.25` |
| `epochs` | `200` |
| Log loss mỗi | `10` hoặc `20` epoch |

Có thể dùng `lr=0.01`, `epochs=300` nếu muốn bảo thủ hơn — miễn **đã scale** continuous.  
**Không** dùng `lr` lớn (vd. 0.1–0.25) khi `transaction_year` còn ~2024 → chắc chắn NaN.

### Metric tham chiếu (doanh số gốc, sau exp) — bản đã chạy ổn

| Metric | ~Giá trị |
|--------|----------|
| MSE | ~8.18e4 |
| RMSE | ~286 |
| MAE | ~174 |
| R² train | ~0.35 |
| R² test | ~0.34 |

R² thang gốc **thấp hơn** R² thang log (bản OLS ~0.70 log) là bình thường vì `exp` phóng đại sai số.

---

## 5. Công thức & skeleton code (copy-paste an toàn)

### 5.1. Scaling — **list feature phải có dấu phẩy đầy đủ**

```python
continuous_features = [
    "discount_pct",
    "unit_price",
    "qty_roll_mean_30d",   # PHẢI có dấu phẩy sau dòng này
    "transaction_year",
    "transaction_month",
    "transaction_day",
    "transaction_dayofweek",
    "customer_age_group_encoded",
]

# Verify tên cột
missing = [c for c in continuous_features if c not in feature_names]
if missing:
    raise ValueError(f"continuous_features không khớp cột: {missing}")

X_mean = np.zeros(len(feature_names))
X_std = np.ones(len(feature_names))
for i, col in enumerate(feature_names):
    if col in continuous_features:
        X_mean[i] = np.mean(X_train[:, i])
        X_std[i] = np.std(X_train[:, i])

X_train = (X_train - X_mean) / (X_std + 1e-8)
X_test = (X_test - X_mean) / (X_std + 1e-8)
```

**Bug kinh điển:** thiếu `,` giữa hai string → Python nối thành  
`'qty_roll_mean_30dtransaction_year'` → không match cột nào → year không scale → **MSE = inf/nan**.

Sau scale: `transaction_year` mean≈0, std≈1, range khoảng `[-1, 1.5]` — **không** còn `[2024, 2025]`.

### 5.2. Class Linear Regression (GD)

```python
class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=750):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        for epoch in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            error = y_pred - y
            dw = (2 / n_samples) * np.dot(X.T, error)
            db = (2 / n_samples) * np.sum(error)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            loss = np.mean(error ** 2)
            self.loss_history.append(loss)
            if epoch % 10 == 0:
                print(f"Epoch {epoch:4d} | MSE = {loss:.6f}")

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
```

### 5.3. Metrics

```python
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
```

### 5.4. Train (một lần)

```python
model = LinearRegression(learning_rate=0.25, epochs=200)
model.fit(X_train, y_train)
# Không fit lại ở bước đánh giá
```

### 5.5. Eval — giải ngược log

```python
predictions_log = model.predict(X_test)
train_predictions_log = model.predict(X_train)

y_test_orig = np.exp(y_test)
y_train_orig = np.exp(y_train)
predictions = np.exp(predictions_log)
train_predictions = np.exp(train_predictions_log)

# Metric trên *_orig và predictions (thang gốc)
# Plot actual vs pred / residual cũng dùng thang gốc
# Loss curve dùng model.loss_history (thang log)
```

---

## 6. Markdown intro (nội dung bắt buộc)

Cell markdown đầu notebook gồm 4 khối:

1. **Giới thiệu bài toán** — dự đoán doanh số; target log; GD from scratch.
2. **Mục tiêu notebook** — pipeline ML, implement GD, scale đúng, eval sau `exp`.
3. **Tổng quan dữ liệu (trước xử lý)** — bảng train/test, 8 continuous + 71 one-hot, missing=0, lưu ý scale.
4. **Mục lục** — map 6 phase kỹ thuật.

Giọng văn: tiếng Việt có dấu, bảng Markdown, ngắn gọn.

---

## 7. Pitfalls đã gặp & cách xử lý

| Triệu chứng | Nguyên nhân | Cách xử lý |
|-------------|-------------|------------|
| MSE epoch 0 ổn (~31), epoch 20 số khổng lồ → `inf` → `nan` | Feature chưa scale (thường `transaction_year`) + `lr` lớn | Scale đủ 8 continuous; kiểm tra range year sau scale |
| In “Scaled 7 features” / preserved 72 | List string bị dính (thiếu `,`) | Viết list nhiều dòng, mỗi tên 1 phần tử; `assert len==8` |
| Mọi metric evaluation = NaN | Weights đã NaN sau train | Fix scale + lr, train lại từ đầu (Restart kernel) |
| Jupyter Run All vẫn NaN dù “đã sửa” | Buffer editor ghi đè file disk / chưa reload | Reload from disk, Restart & Run All |
| R² gốc ~0.35 vs README OLS ~0.70 | Khác thang (gốc vs log) và/hoặc khác solver | So sánh cùng thang; không kết luận nhầm “GD kém OLS” chỉ vì số raw |
| Fit lại lúc eval | Hiểu nhầm “không fit” | Train: `fit` 1 lần; Eval: chỉ `predict` + `exp` |

---

## 8. Checklist nghiệm thu (Definition of Done)

Trước khi coi là “giống phiên bản hiện tại”, tick đủ:

- [ ] Notebook có intro: bài toán + mục tiêu + data overview + mục lục  
- [ ] 6 phase đúng thứ tự (import → load → preprocess → model → train → eval)  
- [ ] `continuous_features` đúng **8** tên, có `transaction_year` và `qty_roll_mean_30d` tách biệt  
- [ ] Sau scale: continuous mean≈0, std≈1; year **không** còn range 2024–2025  
- [ ] One-hot vẫn 0/1  
- [ ] Class GD: `fit` cập nhật `w`,`b`; `loss_history`; `predict`  
- [ ] Train: chỉ `fit` trên train; không NaN trong loss  
- [ ] Loss giảm mượt (vd. ~31 → dưới 1 rồi ổn định)  
- [ ] Eval: `np.exp` rồi mới metric  
- [ ] Có loss curve + actual vs pred + residual  
- [ ] Metric R² test hữu hạn, train ≈ test (không overfitting rõ)  
- [ ] Không dùng OLS/`lstsq` làm solver chính  

---

## 9. Việc **không** làm (trừ khi user yêu cầu)

* Chuyển sang OLS / Ridge / sklearn LR như solver chính  
* Scale cả one-hot  
* Fit mean/std trên test (data leakage)  
* Metric chỉ trên log mà ghi là “doanh số”  
* `fit` lại trong phase đánh giá  
* Xóa kiểm tra `missing` cột continuous  
* Commit / push khi chưa được ask  

---

## 10. Quan hệ với bản OLS (`README.md` / `LinearRegression.ipynb`)

| | OLS | GD (file này / `LR_GD.ipynb`) |
|--|-----|-------------------------------|
| Solver | Normal Equation / `lstsq` | Batch GD lặp epoch |
| Scale | Ít critical hơn | **Bắt buộc** continuous |
| Hội tụ | 1 shot | Phụ thuộc lr, epochs, scale |
| Doc | `README.md` | `LinearRegression.md` + `Gen_LR.md` |

Có thể giữ hai notebook song song; **không trộn** logic cell.

---

## 11. Thứ tự công việc khi agent / người build lại từ zero

1. Đọc `Gen_LR.md` (toàn bộ) + liệt kê file CSV.  
2. Tạo/ghi `LR_GD.ipynb` theo §3 + skeleton §5.  
3. Viết intro markdown §6.  
4. Run All (hoặc exec tuần tự).  
5. Đối chiếu checklist §8; nếu NaN → §7.  
6. (Tuỳ chọn) Cập nhật `LinearRegression.md` cho khớp hyperparams & metric mới.  
7. Không invent data path khác trừ khi user đổi dataset.  

---

## 12. Phiên bản & đồng bộ

| Mục | Giá trị “hiện tại” |
|-----|---------------------|
| Notebook chuẩn | `LR_GD.ipynb` |
| Thuật toán | Linear Regression + Batch GD + MSE |
| LR / epochs | `0.25` / `200` (xem lại cell train nếu đã đổi) |
| Eval | Original scale via `exp` |
| Data | `retail_train_80.csv`, `retail_test_20.csv` |

Khi đổi hyperparams hoặc metric: **cập nhật đồng thời** cell notebook + `LinearRegression.md` + bảng §4 / §12 trong `Gen_LR.md`.

---

## 13. Tóm tắt 30 giây

```text
Data sẵn train/test → tách X/y (sales_amount_log)
→ scale 8 continuous (fit train), giữ one-hot
→ LinearRegression GD: w,b ← MSE gradient, lr=0.25, epochs=200
→ predict → exp → metric + plots trên sales gốc
→ fit chỉ lúc train; NaN = scale/list feature/lr
```

**Đọc file này = đủ context để dựng lại phiên bản Linear GD hiện tại trong chat mới.**
""")
