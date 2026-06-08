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

* **Tên cột:** `text` (nội dung email), `spam` (nhãn: 0 = Non-Spam, 1 = Spam)
* **Kiểm tra cấu trúc:** dùng `df.head()` để xem vài dòng đầu và `df.info()` để kiểm tra kiểu dữ liệu

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
* **Kết quả:** phát hiện một số dòng trùng lặp
* **Xử lý:** xoá bằng `df.drop_duplicates()`

**Tác dụng:** xác nhận dữ liệu không có giá trị rỗng và loại bỏ các dòng trùng lặp để tránh làm sai lệch mô hình.

---

## Phase 3: Exploratory Data Analysis

Mục tiêu: khám phá đặc điểm dữ liệu, phân bố mẫu và các giá trị bất thường trước khi tiền xử lý.

### 3.1 Class Distribution

* Đếm số lượng và tỉ lệ từng lớp bằng `value_counts()`
* Trực quan hóa bằng **Pie Chart**
* **Nhận xét:** dataset không cân bằng hoàn toàn — cần dùng `stratify=y` khi chia train/test

### 3.2 Email Length Analysis

Tạo cột `email_length = df["text"].str.len()` để phân tích độ dài email.

* **Thống kê mô tả:** dùng `describe()` để xem mean, median, std, min, max
* **Group Statistic:** so sánh độ dài trung bình giữa lớp Spam và Non-Spam — email Non-Spam có xu hướng dài hơn

> Độ dài email chỉ dùng cho mục đích phân tích, không đưa vào mô hình.

### 3.3 Histogram Analysis

* Histogram phân bố độ dài email cho toàn bộ dataset
* Phần lớn email tập trung ở vùng độ dài thấp, đuôi kéo dài về phải

### 3.4 Skewness Analysis

* Đo độ lệch của phân phối bằng `df["email_length"].skew()`
* Kết quả cho thấy phân phối lệch phải mạnh — phù hợp với quan sát từ Histogram

### 3.5 Boxplot Analysis

* So sánh phân bố độ dài email giữa hai lớp bằng `sns.boxplot()`
* Cả hai lớp đều có outlier phía trên, Non-Spam có khoảng biến thiên rộng hơn

### 3.6 Outlier Detection

Phát hiện email có độ dài bất thường bằng hai phương pháp:

| Phương pháp | Cách tính |
|-------------|-----------|
| IQR | Lower = Q1 − 1.5×IQR, Upper = Q3 + 1.5×IQR |
| Z-Score | Outlier khi \|z\| > 3 |

* IQR phát hiện nhiều outlier hơn vì phù hợp hơn với phân phối lệch phải
* **Không xoá outlier** — các email dài là dữ liệu thật, giữ lại cho mục đích phân tích

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
* Lưu dữ liệu sạch vào `emails_cleaned.csv`

**Tác dụng:** chuẩn bị dữ liệu văn bản sạch, loại bỏ nhiễu để trích xuất đặc trưng có ý nghĩa.

---

## Phase 5: Dataset Preparation

### Train/Test Split

* **Tỉ lệ:** 80% training, 20% testing
* **Stratified split:** `stratify=y` — đảm bảo tỉ lệ spam/non-spam ở train và test giống tỉ lệ của dataset gốc
* **Reproducibility:** `random_state=42`

### Dataset Export

| File | Nội dung |
|------|----------|
| `train_data.csv` | Tập training (text + spam) |
| `test_data.csv` | Tập testing (text + spam) |

**Tác dụng:** tách dữ liệu để đánh giá công bằng hiệu suất mô hình, tránh data leakage và overfitting.

---

## Phase 6: Feature Engineering

### 6.1 Vocabulary Analysis

* Đếm tổng số từ và số từ duy nhất trong toàn bộ dataset sau khi clean
* Đánh giá mức độ đa dạng từ vựng trước khi vectorization
* Quy mô từ vựng lớn → cần giới hạn bằng `max_features` và các bộ lọc `min_df`, `max_df`

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

### 6.3 Top Informative Terms

Tính Mean TF-IDF của từng token theo từng lớp và trực quan hóa bằng Horizontal Bar Chart:
- **Top 10 Most Informative Terms in Spam Emails**
- **Top 10 Most Informative Terms in Non-Spam Emails**

### Output Files

| File | Nội dung |
|------|----------|
| `emails_cleaned.csv` | Email đã được làm sạch |
| `train_data.csv` | Tập training (text + spam) |
| `test_data.csv` | Tập testing (text + spam) |
| `tfidf_vectorizer.pkl` | Vectorizer đã fit |
| `X_train.pkl` | Ma trận TF-IDF training (sparse matrix) |
| `X_test.pkl` | Ma trận TF-IDF testing (sparse matrix) |
| `y_train.pkl` | Nhãn training |
| `y_test.pkl` | Nhãn testing |

**Tác dụng:** chuyển đổi dữ liệu văn bản thành dạng số thích hợp cho các mô hình Machine Learning (Logistic Regression, SVM, Naive Bayes, Random Forest, ...).
