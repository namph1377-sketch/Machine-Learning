# Phân loại & Tiền xử lý Dữ liệu Email Spam

Tài liệu này giải thích quy trình làm sạch, khám phá, xử lý văn bản và chuẩn bị dữ liệu từ file `Data_P1.ipynb`. Mục tiêu là xử lý bộ dữ liệu email spam, tạo đặc trưng bằng TF-IDF và chuẩn bị dữ liệu để huấn luyện các mô hình phân loại nhị phân (Spam và Non-Spam).

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

Mục tiêu: tìm hiểu tổng quan về tập dữ liệu email trước khi tiến hành phân tích và tiền xử lý.

### 1.1 Load Dataset

* **File dữ liệu:** `emails_raw.csv`
* **Lệnh đọc dữ liệu:** `pd.read_csv()` để đưa dữ liệu vào DataFrame `df`
* **Thư viện sử dụng:** `pandas`, `numpy`, `re`, `matplotlib`, `seaborn`, `sklearn`, `nltk`

### 1.2 Dataset Overview

* **Kích thước ban đầu:** 5.728 dòng, 2 cột
* **Tên cột:**
  - `text`: nội dung email
  - `spam`: nhãn (0 = Non-Spam, 1 = Spam)
* **Kiểm tra:** dùng `df.head()` để xem vài dòng đầu và `df.info()` để kiểm tra kiểu dữ liệu

**Tác dụng:** giúp hiểu cấu trúc dữ liệu trước khi xử lý, phát hiện sớm các kiểu dữ liệu chưa phù hợp.

---

## Phase 2: Data Quality Assessment

Mục tiêu: kiểm tra chất lượng dữ liệu trước khi phân tích.

### 2.1 Missing Values

* **Công cụ:** `df.isnull().sum()`
* **Kết quả:** không có giá trị thiếu ở cả hai cột
* **Xử lý:** nếu có thì xoá bằng `df.dropna()`

### 2.2 Duplicate Values

* **Công cụ:** `df.duplicated().sum()`
* **Kết quả:** phát hiện **33 dòng trùng lặp**
* **Xử lý:** xoá bằng `df.drop_duplicates()`
* **Kích thước sau xử lý:** 5.695 dòng

**Tác dụng:** xác nhận dữ liệu không có giá trị rỗng và loại bỏ các dòng trùng lặp để tránh làm sai lệch mô hình.

---

## Phase 3: Exploratory Data Analysis

Mục tiêu: khám phá đặc điểm dữ liệu, phân bố mẫu và các giá trị bất thường trước khi tiền xử lý.

### 3.1 Class Distribution

* **Công cụ:** `df["spam"].value_counts()`
* **Kết quả:**
  - Non-Spam (0): ~76% dữ liệu
  - Spam (1): ~24% dữ liệu
* **Trực quan hóa:** Pie Chart thể hiện tỉ lệ hai lớp
* **Nhận xét:** Dataset không cân bằng hoàn toàn — cần dùng `stratify=y` khi chia train/test

### 3.2 Email Length Analysis

Tạo cột `email_length = df["text"].str.len()` để phân tích độ dài email.

**Thống kê mô tả:**

| Chỉ số | Giá trị |
|--------|---------|
| Mean | ~1.558 ký tự |
| Median | ~979 ký tự |
| Q1 | ~509 ký tự |
| Q3 | ~1.893 ký tự |
| Max | 43.952 ký tự |
| Std | ~2.047 ký tự |

**So sánh theo lớp:**
* Non-Spam: trung bình ~1.634 ký tự
* Spam: trung bình ~1.317 ký tự
* Email Non-Spam có xu hướng dài hơn email Spam

> Độ dài email chỉ dùng cho mục đích phân tích, không đưa vào mô hình.

### 3.3 Histogram Analysis

* Phần lớn email tập trung ở vùng dưới 2.000 ký tự
* Tần suất giảm mạnh khi độ dài tăng → đuôi kéo dài về phải

### 3.4 Skewness Analysis

* **Kết quả Skewness: ~6.9** → phân phối lệch phải rất mạnh
* Phù hợp với quan sát từ Histogram

### 3.5 Boxplot Analysis

* Cả hai lớp đều có nhiều outlier phía trên
* Non-Spam có khoảng biến thiên rộng hơn và email dài bất thường hơn

### 3.6 Outlier Detection

| Phương pháp | Số outlier phát hiện |
|-------------|----------------------|
| IQR (Q1 - 1.5×IQR, Q3 + 1.5×IQR) | **374** (~6.57%) |
| Z-Score (\|z\| > 3) | **100** (~1.76%) |

* IQR phát hiện nhiều hơn vì phù hợp hơn với phân phối lệch phải
* **Không xoá outlier** — các email dài là dữ liệu thật, chỉ giữ cho mục đích phân tích

**Key Findings:**
- Email Non-Spam có độ dài trung bình lớn hơn Email Spam
- Phân phối lệch phải rất mạnh (Skewness = 6.9)
- Outlier chủ yếu là email có nội dung rất dài, không phải lỗi dữ liệu

---

## Phase 4: Text Preprocessing

Email là dữ liệu văn bản thô nên cần chuẩn hóa trước khi trích xuất đặc trưng.

### Các bước xử lý trong hàm `clean(text)`

| Bước | Xử lý | Lý do |
|------|--------|-------|
| 1 | Chuyển về chữ thường (`text.lower()`) | Tránh coi "FREE", "Free", "free" là 3 từ khác nhau |
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

* Lưu dữ liệu sạch vào `emails_cleaned.csv`
* Xoá email trống sau khi làm sạch
* Xoá email trùng nội dung sau khi chuẩn hóa (phát hiện thêm 200 dòng trùng)
* **Kích thước cuối:** 5.495 email

**Tác dụng:** chuẩn bị dữ liệu văn bản sạch, loại bỏ nhiễu để trích xuất đặc trưng có ý nghĩa.

---

## Phase 5: Dataset Preparation

### Train/Test Split

* **Tỉ lệ:** 80% training, 20% testing
* **Stratified split:** `stratify=y` — đảm bảo tỉ lệ spam/non-spam ở train và test giống tỉ lệ của dataset gốc
* **Reproducibility:** `random_state=42`

| Tập | Số mẫu |
|-----|--------|
| Train | 4.396 |
| Test | 1.099 |

### Dataset Export

| File | Nội dung |
|------|----------|
| `train_data.csv` | Tập training (text + spam) |
| `test_data.csv` | Tập testing (text + spam) |

**Tác dụng:** tách dữ liệu để đánh giá công bằng hiệu suất mô hình, tránh data leakage và overfitting.

---

## Phase 6: Feature Engineering

### 6.1 Vocabulary Analysis

Phân tích quy mô từ vựng trước khi vectorization.

| Chỉ số | Giá trị |
|--------|---------|
| Tổng số từ | 673.961 tokens |
| Số từ duy nhất | 29.561 từ |

* Số lượng từ duy nhất lớn phản ánh sự phong phú của dữ liệu
* Cần giới hạn số đặc trưng bằng `max_features` và các bộ lọc `min_df`, `max_df`

### 6.2 TF-IDF Feature Extraction

Chuyển email từ văn bản thành vector số bằng **TF-IDF** (Term Frequency - Inverse Document Frequency).

**Tham số TfidfVectorizer:**

| Tham số | Giá trị | Lý do |
|---------|---------|-------|
| `token_pattern` | `r'\b\w{2,}\b'` | Chỉ lấy từ có ít nhất 2 ký tự |
| `sublinear_tf` | `True` | Dùng log(tf) — giảm ảnh hưởng từ lặp nhiều lần |
| `min_df` | `5` | Chỉ giữ từ xuất hiện ít nhất 5 emails |
| `max_df` | `0.95` | Loại từ xuất hiện trong hơn 95% emails |
| `ngram_range` | `(1, 2)` | Unigram và bigram |
| `max_features` | `10.000` | Giới hạn số đặc trưng tối đa |

**Quy trình:**
1. Fit + transform trên training data: `vectorizer.fit_transform(X_train_text)`
2. Chỉ transform test data: `vectorizer.transform(X_test_text)` — không fit lại trên test

**Kết quả:**

| Tập | Shape |
|-----|-------|
| X_train | (4.396, 10.000) |
| X_test | (1.099, 10.000) |
| Vocabulary size | 10.000 features |

### 6.3 Top Informative Terms

Tính Mean TF-IDF của từng token theo từng lớp và trực quan hóa bằng Horizontal Bar Chart:
- **Top 10 Most Informative Terms in Spam Emails**
- **Top 10 Most Informative Terms in Non-Spam Emails**

### Output Files

| File | Nội dung |
|------|----------|
| `emails_cleaned.csv` | Email đã được làm sạch |
| `train_data.csv` | Tập training (text + label) |
| `test_data.csv` | Tập testing (text + label) |
| `tfidf_vectorizer.pkl` | Vectorizer đã fit |
| `X_train.pkl` | Ma trận TF-IDF training (sparse matrix) |
| `X_test.pkl` | Ma trận TF-IDF testing (sparse matrix) |
| `y_train.pkl` | Nhãn training |
| `y_test.pkl` | Nhãn testing |

**Tác dụng:** chuyển đổi dữ liệu văn bản thành dạng số thích hợp cho các mô hình Machine Learning (Logistic Regression, SVM, Naive Bayes, Random Forest, ...).
