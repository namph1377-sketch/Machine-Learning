# Hướng dẫn Phân loại & Tiền xử lý Dữ liệu Email Spam

Tài liệu này giải thích quy trình làm sạch, khám phá, xử lý văn bản và chuẩn bị dữ liệu từ file `Data_P1.ipynb`. Mục tiêu của file code là xử lý bộ dữ liệu email spam, tạo đặc trưng bằng TF-IDF và chuẩn bị dữ liệu để huấn luyện các mô hình phân loại nhị phân (Spam và Non-Spam).

---

## Mục lục

1. [Import thư viện và đọc dữ liệu](#1-import-thư-viện-và-đọc-dữ-liệu)  
2. [Data Overview](#2-data-overview)  
3. [Missing Data Analysis và Duplicate](#3-missing-data-analysis-và-duplicate)  
4. [Exploratory Data Analysis - Class Distribution](#4-exploratory-data-analysis---class-distribution)  
5. [Email Length Analysis](#5-email-length-analysis)  
6. [Outlier Detection](#6-outlier-detection)  
7. [Text Preprocessing](#7-text-preprocessing)  
8. [Split train/test](#8-split-traintest)  
9. [Feature Engineering - Word Frequency Analysis](#9-feature-engineering---word-frequency-analysis)  
10. [Vocabulary Analysis](#10-vocabulary-analysis)  
11. [TF-IDF Feature Extraction](#11-tfidf-feature-extraction)  
12. [Lưu dữ liệu sau xử lý](#12-lưu-dữ-liệu-sau-xử-lý)

---

## 1. Import thư viện và đọc dữ liệu

Code bắt đầu bằng việc import các thư viện cần thiết và đọc file dữ liệu email.

* **Thư viện sử dụng:** `pandas`, `numpy`, `re`, `matplotlib.pyplot`, `seaborn`, `sklearn.feature_extraction.text`, `nltk`.
* **File dữ liệu:** `emails_raw.csv`.
* **Lệnh đọc dữ liệu:** dùng `pd.read_csv()` để đưa dữ liệu vào DataFrame.
* **Mục tiêu:** chuẩn bị dữ liệu cho bài toán phân loại email spam và non-spam.

**Tác dụng:** tạo DataFrame ban đầu và chuẩn bị các công cụ xử lý văn bản (NLTK Lemmatizer, TF-IDF Vectorizer) để bắt đầu quá trình kiểm tra, phân tích và tiền xử lý dữ liệu.

---

## 2. Data Overview

Bước này dùng để xem tổng quan cấu trúc ban đầu của dữ liệu.

* **Kích thước dữ liệu:** `df.shape` cho thấy dữ liệu có **5.695 dòng** và **2 cột**.
* **Tên cột:** 
  - `text`: nội dung email
  - `spam`: nhãn (0 = Non-Spam, 1 = Spam)
* **Kiểu dữ liệu:** dùng `df.info()` để kiểm tra cột `text` là kiểu object (chuỗi), cột `spam` là kiểu int (số nguyên).
* **Thống kê mô tả:** dùng `df.head()` để xem các dòng đầu tiên của dataset.

**Tác dụng:** giúp hiểu dữ liệu trước khi xử lý, đồng thời phát hiện sớm các kiểu dữ liệu chưa phù hợp.

---

## 3. Missing Data Analysis và Duplicate

Bước này kiểm tra dữ liệu thiếu và dòng bị trùng lặp.

* **Kiểm tra missing value:** dùng `df.isnull().sum()` để kiểm tra các cột có giá trị rỗng.
* **Kết quả missing:** không có cột nào bị thiếu dữ liệu, nhưng nếu có thì sẽ được xoá bằng `df.dropna()`.
* **Kiểm tra duplicate:** dùng `df.duplicated().sum()` để kiểm tra dòng trùng lặp.
* **Kết quả duplicate:** sau khi xử lý văn bản, sẽ có dòng trùng lặp được phát hiện và xoá bằng `df.drop_duplicates()`.

**Tác dụng:** xác nhận dữ liệu không có giá trị rỗng, và loại bỏ các dòng trùng lặp để tránh làm sai lệch mô hình.

---

## 4. Exploratory Data Analysis - Class Distribution

Bước này phân tích sự phân bố giữa hai lớp email.

* **Kiểm tra nhãn:** dùng `df["spam"].value_counts()` để đếm số email spam và non-spam.
* **Kết quả phân bố:**
  - **Số lượng Non-Spam:** khoảng 70-75% dữ liệu.
  - **Số lượng Spam:** khoảng 25-30% dữ liệu.
* **Trực quan hóa:** dùng Pie Chart để thể hiện tỉ lệ giữa hai lớp.
* **Nhận xét:** Dataset không cân bằng hoàn toàn, nhưng sự chênh lệch không quá lớn.

**Tác dụng:** hiểu được tỉ lệ Spam/Non-Spam trong dữ liệu, từ đó có thể cân nhắc việc sử dụng stratified split khi chia train/test.

---

## 5. Email Length Analysis

Bước này tạo đặc trưng độ dài email và phân tích phân phối.

* **Feature Engineering:** tạo cột `email_length = df["text"].str.len()` để tính độ dài mỗi email.
* **Thống kê mô tả:**
  * Độ dài trung bình: khoảng **1.558 ký tự**
  * Trung vị (Median): khoảng **979 ký tự**
  * Độ dài tối thiểu: **509 ký tự** (Q1)
  * Độ dài tối đa: **43.952 ký tự** (Q3 = 1.893)
  * Độ lệch chuẩn: **2.047 ký tự**
* **So sánh giữa hai lớp:** 
  * Email Non-Spam có độ dài trung bình **cao hơn** Email Spam
  * Email Non-Spam: trung bình ~1.800 ký tự
  * Email Spam: trung bình ~1.000 ký tự
* **Phân tích Skewness:** Giá trị Skewness khoảng **6.9**, cho thấy phân phối **lệch phải rất mạnh**
* **Trực quan hóa:** Histogram và Boxplot cho thấy phần lớn email tập trung ở vùng độ dài thấp

**Tác dụng:** giúp hiểu đặc điểm cấu trúc dữ liệu, phát hiện khác biệt giữa spam và non-spam dựa trên độ dài email.

---

## 6. Outlier Detection

Bước này phát hiện email có độ dài bất thường.

* **Phương pháp IQR (Interquartile Range):**
  * Tính Q1, Q3, IQR = Q3 - Q1
  * Lower bound = Q1 - 1.5 × IQR
  * Upper bound = Q3 + 1.5 × IQR
  * **Kết quả:** Phát hiện **374 outlier**, chiếm khoảng **6.57%** dữ liệu
* **Phương pháp Z-Score:**
  * Tính z-score cho mỗi giá trị: z = (x - mean) / std
  * Xác định outlier khi |z| > 3
  * **Kết quả:** Phát hiện **100 outlier**, chiếm khoảng **1.76%** dữ liệu
* **Cách xử lý:** Chỉ phát hiện và phân tích, **không xoá outlier** vì các email dài có thể là dữ liệu thật

**Tác dụng:** giúp nhận biết rằng tồn tại các email có độ dài rất khác biệt, nhưng chúng được giữ lại vì có giá trị thực tế.

---

## 7. Text Preprocessing

Email là dữ liệu văn bản thô nên cần được chuẩn hóa trước khi trích xuất đặc trưng.

Các bước xử lý:

1. **Chuyển về chữ thường:** `text.lower()` - chuẩn hóa tất cả thành chữ thường (Tránh coi Free, FREE, free là 3 từ khác nhau).
2. **Loại bỏ tiền tố:** Xoá "RE:", "FW:", "FWD:" - các tiền tố email trả lời/chuyển tiếp (Vì các tiền tố email không giúp phân loại).
3. **Loại bỏ "Subject:":** Xoá các dòng chủ đề email (Vì dòng chủ đề không giúp phân loại)
4. **Loại bỏ URL:** `https?://\S+|www\.\S+` - xoá các liên kết web (Vì URL thường dài, đa dạng và gây nhiễu cho mô hình).
5. **Loại bỏ HTML tags:** `<.*?>` - xoá các thẻ HTML (Vì nó là mã định dạng chứ không phải nội dung văn bản).
6. **Loại bỏ ký tự không phải chữ cái:** Giữ chỉ chữ cái a-z và khoảng trắng (Giảm nhiễu và làm dữ liệu đồng nhất hơn).
7. **Dọn khoảng trắng thừa:** Loại bỏ khoảng trắng dư thừa và strip đầu/cuối (Tránh tạo token rỗng hoặc lỗi khi tách từ).
8. **Loại bỏ Stopword:** Xoá các từ phổ biến tiếng Anh (Vì các từ này xuất hiện quá nhiều nhưng ít giá trị phân biệt spam).
9. **Lemmatization:** Chuyển động từ về dạng gốc (running → run) (Giảm số chiều dữ liệu và tăng tần suất xuất hiện từ).
10. **Giữ từ dài >= 3 ký tự:** Loại bỏ từ quá ngắn để tập trung vào từ có nghĩa (Giảm nhiễu từ các từ ít thông tin như ok, hi, at,...).

**Kết quả sau xử lý:**
* Xoá email trống (blank) sau khi làm sạch
* Xoá các email trùng lặp về nội dung sau khi chuẩn hóa
* Lưu dữ liệu sạch vào `emails_cleaned.csv`

**Tác dụng:** chuẩn bị dữ liệu văn bản sạch, loại bỏ nhiễu, để trích xuất đặc trưng có ý nghĩa cho mô hình.

---

## 8. Split train/test

Dữ liệu được chia thành tập huấn luyện và tập kiểm tra.

* **Cách chia:** 80% dữ liệu làm training, 20% dữ liệu làm testing.
* **Sử dụng stratified split:** `stratify=y` đảm bảo tỉ lệ spam/non-spam ở train và test giống tỉ lệ của dataset gốc.
* **Cố định random_state:** `random_state=42` để đảm bảo tính lặp lại (reproducibility) khi chia dữ liệu.
* **Train shape:** `(4556, )` - số lượng mẫu train (chứa cột text)
* **Test shape:** `(1139, )` - số lượng mẫu test
* **Không sử dụng random split:** vì mục đích của bài toán là phân loại email mà không liên quan đến thứ tự thời gian

**Tác dụng:** tách dữ liệu để đánh giá công bằng hiệu suất mô hình, tránh overfitting.

---

## 9. Feature Engineering - Word Frequency Analysis

Bước này phân tích các từ xuất hiện nhiều nhất trong mỗi lớp email.

* **Top 10 từ trong Spam Emails:**
  * Xác định các từ khóa đặc trưng của email spam
  * Ví dụ: "click", "prize", "winner", "free", "offer"
  * Trực quan hóa bằng Horizontal Bar Chart

* **Top 10 từ trong Non-Spam Emails:**
  * Xác định các từ khóa đặc trưng của email hợp lệ
  * Ví dụ: "meeting", "report", "project", "team", "data"
  * Trực quan hóa bằng Horizontal Bar Chart

* **Mục tiêu:** Hiểu các từ vựng khác biệt giữa hai lớp, hỗ trợ phân tích dữ liệu (word frequency có thể được sử dụng làm feature ngoài TF-IDF).

**Tác dụng:** cung cấp insight về đặc điểm văn bản của spam vs. non-spam, giúp nhận biết các từ mang tính phân biệt cao.

---

## 10. Vocabulary Analysis

Bước này phân tích quy mô từ vựng của tập dữ liệu.

* **Tổng số từ:** Số lượng token (từ) sau khi tách và chuẩn hóa.
* **Số từ duy nhất:** Số lượng từ vựng duy nhất trong toàn bộ dataset.
* **Mục tiêu:** Đánh giá mức độ đa dạng và độ phức tạp của dữ liệu văn bản.

**Kết quả từ notebook:**
* **Tổng số từ:** Lớn (phần lớn token từ tập dữ liệu)
* **Số từ duy nhất:** 29.101 từ

**Nhận xét:**
* Số lượng từ duy nhất lớn phản ánh sự phong phú của dữ liệu văn bản
* Cung cấp cơ sở cho TF-IDF nhằm trích xuất các đặc trưng có giá trị phân biệt
* Quy mô từ vựng lớn sẽ dẫn đến số chiều dữ liệu cao sau vectorization, cần giới hạn bằng tham số `max_features` và các bộ lọc min_df, max_df

**Tác dụng:** hiểu được độ phức tạp của dữ liệu văn bản trước khi trích xuất đặc trưng.

---

## 11. TF-IDF Feature Extraction

Bước này chuyển email từ văn bản thành vector số bằng TF-IDF (Term Frequency - Inverse Document Frequency).

* **TF-IDF là gì:**
  * **TF (Term Frequency):** Tần suất từ xuất hiện trong tài liệu
  * **IDF (Inverse Document Frequency):** Trọng số phản ánh tính hiếm có của từ (từ hiếm → trọng số cao)
  * Tích của TF và IDF cho mỗi từ trong mỗi email

* **Tham số chính:**
  * `max_features=10000` - Giữ tối đa 10.000 từ quan trọng nhất
  * `min_df=5` - Chỉ giữ từ xuất hiện ít nhất 5 emails (loại bỏ từ quá hiếm)
  * `max_df=0.95` - Loại bỏ từ xuất hiện trong hơn 95% emails (quá phổ biến, ít giá trị phân biệt)
  * `ngram_range=(1,2)` - Dùng unigram (1 từ) và bigram (2 từ liên tiếp)
  * `token_pattern=r'\b\w{3,}\b'` - Chỉ giữ từ có ít nhất 3 ký tự

* **Quy trình:**
  1. Fit vectorizer trên training data: `vectorizer.fit_transform(X_train_text)`
  2. Transform test data: `vectorizer.transform(X_test_text)` (không fit lại trên test)
  3. Kết quả là sparse matrix (ma trận thưa)

* **Kết quả sau vectorization:**
  * Train shape: `(4393, 10000)` - 4393 email, mỗi email được biểu diễn bằng 10000 chiều
  * Test shape: `(1099, 10000)` - tương ứng với test data
  * Vocabulary size ban đầu: 29.101 từ duy nhất
  * Vocabulary size sau lọc: 10.000 từ (giới hạn bằng `max_features=10000`)

**Tác dụng:** chuyển đổi dữ liệu văn bản thành dạng số thích hợp cho các mô hình Machine Learning.

---

## 12. Lưu dữ liệu sau xử lý

Bước cuối cùng là lưu các dữ liệu đã xử lý thành file để sử dụng cho các mô hình tiếp theo.

* **File được lưu:**
  1. `emails_cleaned.csv` - Email đã được làm sạch (văn bản gốc sau preprocessing)
  2. `train_data.csv` - Tập training với cột text và spam
  3. `test_data.csv` - Tập test với cột text và spam
  4. `tfidf_vectorizer.pkl` - Vectorizer đã fit, dùng để transform dữ liệu mới
  5. `X_train.pkl` - Ma trận TF-IDF của training data (sparse matrix)
  6. `X_test.pkl` - Ma trận TF-IDF của test data (sparse matrix)
  7. `y_train.pkl` - Nhãn của training data
  8. `y_test.pkl` - Nhãn của test data

* **Mục tiêu:** Chuẩn bị dữ liệu đã được xử lý, sạch và vector hóa, sẵn sàng cho các mô hình phân loại (Logistic Regression, SVM, Naive Bayes, Random Forest, v.v.).

**Tác dụng:** tạo dữ liệu sạch, đã được vectorization, để sử dụng cho huấn luyện và đánh giá các mô hình học máy.