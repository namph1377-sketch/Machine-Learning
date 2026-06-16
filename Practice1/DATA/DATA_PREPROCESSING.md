# CLASSIFYING SPAM EMAILS — Data Preprocessing

Tài liệu mô tả bài toán, nguồn dữ liệu và quy trình tiền xử lý email phục vụ phân loại **Spam / Non-Spam**. Pipeline chính được triển khai trong `Balance_Data/Balance_Data.ipynb`.

---

## 1. Bài toán

**Phân loại email nhị phân:** xác định một email là **Spam** (1) hay **Non-Spam / Ham** (0) dựa trên nội dung văn bản `text`.

Đây là bài toán **Text Classification** trong lĩnh vực NLP, ứng dụng thực tế trong lọc thư rác, bảo vệ hộp thư và giảm nhiễu thông tin.

---

## 2. Mục tiêu

| Mục tiêu | Mô tả |
|----------|--------|
| Hiểu dữ liệu | Khám phá phân bố nhãn, độ dài email, chất lượng dữ liệu |
| Xử lý mất cân bằng | Từ tập gốc lệch về ham → tạo tập cân bằng hơn để train model công bằng hơn |
| Làm sạch văn bản | Chuẩn hóa email thô (URL, HTML, stopwords, lemmatization...) |
| Trích xuất đặc trưng | Chuyển text → vector TF-IDF |
| Chuẩn bị huấn luyện | Chia train/test, export CSV và file `.pkl` cho các mô hình ML |

---

## 3. Giải pháp tổng quan

```text
[Thu thập dữ liệu]
        ↓
[Tập gốc mất cân bằng: emails_raw.csv]
        ↓
[Bổ sung 2.700 email spam từ email_raw1.csv]
        ↓
[Gộp + loại trùng → emails_balanced.csv]
        ↓
[EDA → Preprocessing → Train/Test Split → TF-IDF]
        ↓
[Output: balance_emails_cleaned.csv, train_balance.csv, *.pkl]
        ↓
[Huấn luyện mô hình: Logistic Regression, Naive Bayes, ...]
```

**Ý tưởng chính:** không train trực tiếp trên tập lệch nhãn nặng (~76% ham / ~24% spam), mà **bổ sung spam có kiểm soát** để hai lớp gần cân bằng (~48% / ~52%), sau đó mới clean và vectorize.

---

## 4. Thu thập dữ liệu

### 4.1 Nguồn dữ liệu

| File | Vai trò | Cột | Số lượng (thô) |
|------|---------|-----|----------------|
| `emails_raw.csv` | Tập gốc | `text`, `spam` | 5.728 email |
| `email_raw1.csv` | Tập bổ sung | `label`, `text` | 28.063 email |
| `emails_balanced.csv` | Tập sau gộp | `text`, `spam` | 8.395 email |

**Phân bố nhãn tập thô:**

| Tập | Tổng | Spam (1) | Non-Spam (0) | Ghi chú |
|-----|------|----------|--------------|---------|
| `emails_raw.csv` | 5.728 | 1.368 (23,9%) | 4.360 (76,1%) | **Mất cân bằng** — ham chiếm đa số |
| `email_raw1.csv` | 28.063 | 13.776 (49,1%) | 14.287 (50,9%) | Tập lớn, gần cân bằng; chỉ lấy phần spam |

> **Nguồn Kaggle (điền link sau):**
> - Dataset gốc `emails_raw.csv`: `[LINK_KAGGLE_EMAILS_RAW]`
> - Dataset bổ sung `email_raw1.csv`: `[LINK_KAGGLE_EMAIL_RAW1]`

### 4.2 Cân bằng nhãn — bước đầu xử lý

Thực hiện trong **Phase 1 — Load Dataset** của `Balance_Data/Balance_Data.ipynb`:

```python
# 1. Đọc 2 tập thô
df_raw = pd.read_csv("emails_raw.csv")
df_extra = pd.read_csv("email_raw1.csv")
df_extra = df_extra.rename(columns={"label": "spam"})

# 2. Lấy 2.700 email spam ngẫu nhiên từ tập bổ sung
spam_sample = df_extra[df_extra["spam"] == 1].sample(n=2700, random_state=42)

# 3. Gộp với tập gốc
df = pd.concat([df_raw, spam_sample], ignore_index=True)

# 4. Loại trùng theo nội dung text
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

# 5. Lưu tập cân bằng
df.to_csv("emails_balanced.csv", index=False)
```

**Kết quả sau gộp:**

| Giai đoạn | Tổng | Spam | Non-Spam | Tỷ lệ spam |
|-----------|------|------|----------|------------|
| Trước gộp (`emails_raw.csv`) | 5.728 | 1.368 | 4.360 | 23,9% |
| Sau gộp (trước loại trùng) | 8.428 | 4.068 | 4.360 | 48,3% |
| `emails_balanced.csv` (sau loại trùng) | **8.395** | **4.068** | **4.327** | **48,5%** |

→ Từ tập **lệch ham ~76%** thành tập **gần cân bằng ~48,5% spam / ~51,5% ham**, loại **33 email** trùng nội dung khi gộp.

---

## 5. EDA (khám phá dữ liệu) — tóm tắt

Thực hiện trên `emails_balanced.csv` trước bước clean text.

| Bước | Nội dung | Nhận xét chính |
|------|----------|----------------|
| **3.1 Class Distribution** | Đếm & pie chart spam/ham | Hai lớp gần cân bằng sau bước gộp |
| **3.2 Email Length** | `email_length`, mean/median theo lớp | Ham dài hơn spam (~1.634 vs ~1.322 ký tự); median 871 |
| **3.3 Histogram** | Phân phối độ dài | Phần lớn email < 2.000 ký tự, đuôi dài về phải |
| **3.4 Skewness** | `df["email_length"].skew()` | Skewness ≈ 6.0 — lệch phải mạnh |
| **3.5 Boxplot** | So sánh spam vs ham | Cả hai lớp có outlier phía trên |
| **3.6 Outlier** | IQR & Z-Score | IQR: 594 outliers; Z-Score: 161 outliers — **không xóa** |

> Độ dài email chỉ phục vụ phân tích, **không** đưa vào mô hình.

---

## 6. Tiền xử lý văn bản (Text Preprocessing)

Hàm `clean(text)` áp dụng cho toàn bộ `df["text"]`:

| # | Bước | Mục đích |
|---|------|----------|
| 1 | Lowercase | Thống nhất chữ hoa/thường |
| 2 | Xóa tiền tố `re:`, `fw:`, `fwd:` | Loại metadata reply/forward |
| 3 | Xóa `subject:` | Loại dòng chủ đề |
| 4 | Xóa URL (`http`, `www`) | Giảm nhiễu từ link |
| 5 | Xóa HTML tags | Chỉ giữ nội dung text |
| 6 | Xóa ký tự không phải chữ cái | Chuẩn hóa token |
| 7 | Dọn khoảng trắng | Tránh token rỗng |
| 8 | Stopwords (EN + `com`, `www`, `net`, `org`) | Bỏ từ phổ biến |
| 9 | Lemmatization | Gộp biến thể động từ |
| 10 | Giữ từ > 1 ký tự | Loại token quá ngắn |

**Sau clean:**
- Xóa email trống và trùng nội dung
- Lưu `balance_emails_cleaned.csv` — **8.191 email**

---

## 7. Chuẩn bị tập train/test

| Tham số | Giá trị |
|---------|---------|
| Tỉ lệ chia | 80% train / 20% test |
| Stratify | `stratify=y` — giữ tỷ lệ spam/ham |
| Random seed | `random_state=42` |

| Tập | Số mẫu |
|-----|--------|
| Train | 6.552 |
| Test | 1.639 |

**File export:** `train_balance.csv`, `test_balance.csv`

---

## 8. Feature Engineering — TF-IDF

| Tham số | Giá trị |
|---------|---------|
| `token_pattern` | `\b\w{2,}\b` |
| `sublinear_tf` | `True` |
| `min_df` | `3` |
| `max_df` | `0.95` |
| `ngram_range` | `(1, 2)` |
| `max_features` | `10.000` |

**Từ vựng sau clean:** ~1.008.331 token, ~60.093 từ unique. Giới hạn 10.000 feature vẫn giữ ~91% tổng tần suất từ.

**File output:**

| File | Mô tả |
|------|--------|
| `balance_emails_cleaned.csv` | Email đã clean |
| `train_balance.csv` / `test_balance.csv` | Tập train/test |
| `tfidf_vectorizer_balance.pkl` | Vectorizer đã fit |
| `X_train_balance.pkl` / `X_test_balance.pkl` | Ma trận TF-IDF |
| `y_train_balance.pkl` / `y_test_balance.pkl` | Nhãn |

---

## 9. Workflow đầy đủ

```text
Phase 1: Data Understanding
    ├── Load emails_raw.csv + email_raw1.csv
    ├── Thống kê 2 tập thô
    ├── Lấy 2.700 spam → gộp → emails_balanced.csv
    └── Dataset Overview

Phase 2: Data Quality Assessment
    ├── Missing values
    └── Duplicate rows

Phase 3: Exploratory Data Analysis
    ├── Class distribution
    ├── Email length (stats, histogram, skewness, boxplot)
    └── Outlier detection (IQR, Z-Score)

Phase 4: Text Preprocessing
    ├── Prefix analysis (re/fw/subject)
    └── clean() → balance_emails_cleaned.csv

Phase 5: Dataset Preparation
    └── Train/Test split → train_balance.csv, test_balance.csv

Phase 6: Feature Engineering
    ├── Vocabulary analysis & cumulative coverage
    ├── TF-IDF vectorization
    └── Top informative terms (spam vs ham)
```

---

## 10. Notebook tham chiếu

| Notebook | Mô tả |
|----------|--------|
| `Data_P1 (1).ipynb` | Pipeline trên tập gốc `emails_raw.csv` (không cân bằng) |
| `Balance_Data/Balance_Data.ipynb` | Pipeline đầy đủ có bước cân bằng nhãn + tiền xử lý |

---

## 11. Ghi chú

- `random_state=42` được dùng xuyên suốt để kết quả tái lập được.
- Một số token metadata (ví dụ `cc`, `hou`, `ect`) có thể sót sau clean — xem phân tích trong notebook.
- Link Kaggle cần bổ sung tại mục [4.1](#41-nguồn-dữ-liệu) trước khi nộp báo cáo.