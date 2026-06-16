# Phân loại & Tiền xử lý Dữ liệu Email Spam — Balanced Dataset

Tài liệu này giải thích quy trình xử lý dữ liệu từ file `Balance_Data.ipynb`. So với `Data_P1.ipynb`, notebook này bổ sung bước **cân bằng dataset** bằng cách gộp thêm email spam từ nguồn dữ liệu ngoài trước khi thực hiện các bước phân tích và tiền xử lý.

---

## Mục lục

1. [Phase 1: Data Understanding](#phase-1-data-understanding)
2. [Phase 2: Data Quality Assessment](#phase-2-data-quality-assessment)
3. [Phase 3: Exploratory Data Analysis](#phase-3-exploratory-data-analysis)
4. [Phase 4: Text Preprocessing](#phase-4-text-preprocessing)
5. [Phase 5: Dataset Preparation](#phase-5-dataset-preparation)
6. [Phase 6: Feature Engineering](#phase-6-feature-engineering)

---

## Phase 1: Data Understanding

Mục tiêu: tải và gộp dữ liệu từ hai nguồn, sau đó tìm hiểu tổng quan tập dữ liệu đã cân bằng.

### 1.1 Load Dataset

Notebook đọc và gộp dữ liệu từ hai file:

| File | Nội dung |
|------|----------|
| `emails_raw.csv` | Dataset gốc (spam + ham) |
| `email_raw1.csv` | Dataset bổ sung (lấy thêm spam) |

**Quy trình gộp:**
1. Đổi tên cột `label` → `spam` trong file bổ sung để đồng nhất format
2. Lấy 2.700 email spam từ file bổ sung (`sample(n=2700, random_state=42)`)
3. Gộp với dataset gốc bằng `pd.concat()`
4. Xoá trùng lặp theo cột `text`
5. Lưu ra `emails_balanced.csv`

> Mục đích: giảm sự mất cân bằng giữa lớp spam và non-spam so với dataset gốc.

### 1.2 Dataset Overview

* Dùng `df.head()` và `df.info()` để kiểm tra cấu trúc sau khi gộp
* **Cột:** `text` (nội dung email), `spam` (nhãn: 0 = Non-Spam, 1 = Spam)

---

## Phase 2: Data Quality Assessment

Mục tiêu: kiểm tra chất lượng dữ liệu sau khi gộp.

### 2.1 Missing Values

* Kiểm tra bằng `df.isnull().sum()`
* Xoá nếu có bằng `df.dropna()`

### 2.2 Duplicate Values

* Kiểm tra bằng `df.duplicated().sum()`
* Xoá bằng `df.drop_duplicates()`

**Tác dụng:** đảm bảo dữ liệu sạch trước khi phân tích.

---

## Phase 3: Exploratory Data Analysis

Mục tiêu: khám phá đặc điểm dữ liệu sau khi cân bằng.

### 3.1 Class Distribution

* Đếm số lượng và tỉ lệ từng lớp bằng `value_counts()`
* Trực quan hóa bằng **Pie Chart**
* Nhận xét: tỉ lệ spam/non-spam sau khi gộp cân bằng hơn so với dataset gốc

### 3.2 Email Length Analysis

Tạo cột `email_length = df["text"].str.len()` để phân tích độ dài email.

* **Thống kê mô tả:** mean, median, std, min, max
* **Group Statistic:** so sánh độ dài trung bình giữa lớp Spam và Non-Spam

> Độ dài email chỉ dùng cho mục đích phân tích, không đưa vào mô hình.

### 3.3 Histogram Analysis

* Histogram phân bố độ dài email cho toàn bộ dataset
* Quan sát hình dạng phân phối và mức độ tập trung dữ liệu

### 3.4 Skewness Analysis

* Đo độ lệch của phân phối bằng `df["email_length"].skew()`
* Skewness > 0: phân phối lệch phải

### 3.5 Boxplot Analysis

* So sánh phân bố độ dài email giữa hai lớp
* Phát hiện outlier trực quan

### 3.6 Outlier Detection

Phát hiện email có độ dài bất thường bằng hai phương pháp:

| Phương pháp | Cách tính |
|-------------|-----------|
| IQR | Lower = Q1 − 1.5×IQR, Upper = Q3 + 1.5×IQR |
| Z-Score | Outlier khi \|z\| > 3 |

* **Không xoá outlier** — các email dài là dữ liệu thật, giữ lại để phân tích

---

## Phase 4: Text Preprocessing

Email là dữ liệu văn bản thô nên cần chuẩn hóa trước khi trích xuất đặc trưng.

### Các bước xử lý trong hàm `clean(text)`

| Bước | Xử lý | Lý do |
|------|--------|-------|
| 1 | Chuyển về chữ thường | Tránh coi "FREE", "Free", "free" là 3 từ khác nhau |
| 2 | Loại bỏ tiền tố RE/FW/FWD | Tiền tố không giúp phân loại |
| 3 | Loại bỏ "Subject:" | Dòng chủ đề không mang thông tin phân loại |
| 4 | Loại bỏ URL | URL dài, đa dạng và gây nhiễu cho mô hình |
| 5 | Loại bỏ HTML tags | Mã định dạng, không phải nội dung văn bản |
| 6 | Loại bỏ ký tự không phải chữ cái | Giảm nhiễu và đồng nhất dữ liệu |
| 7 | Dọn khoảng trắng thừa | Tránh tạo token rỗng |
| 8 | Loại bỏ Stopwords (ENGLISH_STOP_WORDS + custom) | Từ phổ biến ít giá trị phân biệt spam |
| 9 | Lemmatization (động từ → dạng gốc) | Giảm chiều dữ liệu, tăng tần suất từ |
| 10 | Giữ từ dài > 1 ký tự | Loại bỏ từ quá ngắn, ít thông tin |

**Custom Stopwords bổ sung:** `com`, `www`, `net`, `org`

### Kết quả sau preprocessing

* Xoá email trống và email trùng nội dung sau khi chuẩn hóa
* Lưu dữ liệu sạch vào `balance_emails_cleaned.csv`

---

## Phase 5: Dataset Preparation

### Train/Test Split

* **Tỉ lệ:** 80% training, 20% testing
* **Stratified split:** `stratify=y` — giữ nguyên tỉ lệ spam/non-spam ở cả hai tập
* **Reproducibility:** `random_state=42`

### Dataset Export

| File | Nội dung |
|------|----------|
| `train_balance.csv` | Tập training (text + spam) |
| `test_balance.csv` | Tập testing (text + spam) |

---

## Phase 6: Feature Engineering

### 6.1 Vocabulary Analysis

* Đếm tổng số từ và số từ duy nhất trong toàn bộ dataset sau khi clean
* Đánh giá mức độ đa dạng từ vựng trước khi vectorization

### 6.2 TF-IDF Feature Extraction

Chuyển email từ văn bản thành vector số bằng **TF-IDF**.

**Tham số TfidfVectorizer:**

| Tham số | Giá trị | Lý do |
|---------|---------|-------|
| `token_pattern` | `r'\b\w{2,}\b'` | Chỉ lấy từ có ít nhất 2 ký tự |
| `sublinear_tf` | `True` | Dùng log(tf) — giảm ảnh hưởng từ lặp nhiều lần |
| `min_df` | `3` | Chỉ giữ từ xuất hiện ít nhất 3 emails |
| `max_df` | `0.95` | Loại từ xuất hiện trong hơn 95% emails |
| `ngram_range` | `(1, 2)` | Unigram và bigram |
| `max_features` | `10.000` | Giới hạn số đặc trưng tối đa |

**Quy trình:**
1. Fit + transform trên training data: `vectorizer.fit_transform(X_train_text)`
2. Chỉ transform test data: `vectorizer.transform(X_test_text)` — không fit lại trên test

### 6.3 Top Informative Terms

Tính Mean TF-IDF của từng token theo từng lớp và trực quan hóa bằng Horizontal Bar Chart:
- **Top 10 Most Informative Terms in Spam Emails**
- **Top 10 Most Informative Terms in Non-Spam Emails**

### Output Files

| File | Nội dung |
|------|----------|
| `balance_emails_cleaned.csv` | Email đã được làm sạch |
| `train_balance.csv` | Tập training (text + label) |
| `test_balance.csv` | Tập testing (text + label) |
| `tfidf_vectorizer_balance.pkl` | Vectorizer đã fit |
| `X_train_balance.pkl` | Ma trận TF-IDF training (sparse matrix) |
| `X_test_balance.pkl` | Ma trận TF-IDF testing (sparse matrix) |
| `y_train_balance.pkl` | Nhãn training |
| `y_test_balance.pkl` | Nhãn testing |

**Tác dụng:** chuyển đổi dữ liệu văn bản thành dạng số thích hợp cho các mô hình Machine Learning (Logistic Regression, SVM, Naive Bayes, ...).
