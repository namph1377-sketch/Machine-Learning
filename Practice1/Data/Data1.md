# Hướng dẫn sử dụng Notebook `Data_Practice1.ipynb`

Tập tin này mô tả các bước thực thi trong `Data_Practice1.ipynb` (xử lý bộ dữ liệu email spam `emails_raw.csv`) và các artifact mà notebook tạo ra nhằm chuẩn bị dữ liệu phục vụ bài toán phân loại email spam và non-spam.

---

## Mục lục

1. [Tóm tắt nhanh](#tóm-tắt-nhanh)
2. [Thư viện chính](#thư-viện-chính-được-import-trong-notebook)
3. [Luồng xử lý](#luồng-xử-lý-tương-ứng-các-cell-trong-notebook)

---

## Tóm tắt nhanh

- File dữ liệu đầu vào: `emails_raw.csv` (các cột quan trọng: `text`, `spam`).
- Mục tiêu: tiền xử lý văn bản (clean → remove stopwords → lemmatize), EDA, chia train/test, vectorize bằng TF‑IDF và lưu artifact.
- Artifact đầu ra (được lưu bởi notebook):
  - `emails_cleaned.csv` (sau làm sạch),
  - `train_data.csv`, `test_data.csv` (text + spam, chưa vector),
  - `tfidf_vectorizer.pkl`, `X_train.pkl`, `X_test.pkl`, `y_train.pkl`, `y_test.pkl`.


## Thư viện chính (được import trong notebook)

- `pandas`, `numpy`, `re`, `matplotlib`, `seaborn`, `scikit-learn` (`TfidfVectorizer`), `joblib`, `nltk` (WordNet lemmatizer)


## Luồng xử lý (tương ứng các cell trong notebook)

1. Load dữ liệu và kiểm tra cấu trúc (`df.head()`, `df.info()`, `df.isnull().sum()`).
2. Data quality: loại bỏ dòng null và duplicate bằng `df.dropna()` / `df.drop_duplicates()`.
3. Text preprocessing — hàm `clean()`:
   - lowercase; xoá các tiền tố trả lời (`re`, `fw`, `fwd`); xoá `subject:` và URL; loại HTML tags; xoá các kí tự đặc biệt và dọn khoảng trắng thừa;
   - loại stopwords tiếng Anh bằng `ENGLISH_STOP_WORDS`;
   - lemmatize động từ bằng `WordNetLemmatizer()` (ví dụ `studying` → `study`, `understood` → `understand`);
   - lưu kết quả `df.to_csv('emails_cleaned.csv', index=False)`.
4. EDA:
   - Phân bố nhãn `spam` (bar plot), `email_length` (describe, histogram, boxplot), heatmap tương quan (`email_length` vs `spam`).
   - Top‑20 từ xuất hiện nhiều nhất trong `spam` và `not spam` (đếm từ thô sau cleaning) — hai biểu đồ riêng cho spam và ham.
5. Train/test split:
   - `train_test_split(..., test_size=0.2, random_state=42, stratify=y)`;
   - Lưu `train_data.csv` và `test_data.csv` (chưa vector).
6. TF‑IDF vectorization:
   - Tham số notebook: `TfidfVectorizer(max_features=10000, min_df=2, ngram_range=(1,2), token_pattern=r"\\b\\w{4,}\\b")`.
   - `X_train = vectorizer.fit_transform(X_train_text)`; `X_test = vectorizer.transform(X_test_text)`.
   - Lưu `tfidf_vectorizer.pkl`, `X_train.pkl`, `X_test.pkl`, `y_train.pkl`, `y_test.pkl`.
