# Machine Learning Practice

Repository chứa các bài thực hành môn Machine Learning, bao gồm các bài toán phân loại và hồi quy được xây dựng từ đầu bằng Python.

---

## 📂 Cấu trúc thư mục

```text
Machine_Learning_Practice/
│
├── Practice1/
│   ├── DATA/
│   ├── MODEL/
│   │   ├── LinearRegression/
│   │   ├── NaiveBayes/
│   │   └── SVM/
│   └── README.md
│
├── Practice2/
│   ├── DATA/
│   │   ├── retail_sales_dataset.csv
│   │   ├── retail_train_80.csv
│   │   └── retail_test_20.csv
│   │
│   ├── Model/
│   │   ├── LinearRegression/
│   │   ├── DecisionTree/
│   │   └── RandomForest/
│   └── README.md
│
└── README.md
```

---

# Practice 1: Email Spam Classification

## Mục tiêu

Xây dựng và đánh giá các mô hình học máy để phân loại email thành:

- Spam (thư rác)
- Ham (thư hợp lệ)

## Tiền xử lý dữ liệu

- Chuyển văn bản về chữ thường
- Loại bỏ ký tự đặc biệt
- Tokenization
- TF-IDF Vectorization

## Các mô hình sử dụng

### 1. Linear Regression

Sử dụng hồi quy tuyến tính cho bài toán phân loại nhị phân bằng cách áp dụng ngưỡng (threshold) trên giá trị dự đoán.

### 2. Naive Bayes

Mô hình xác suất dựa trên định lý Bayes với giả định độc lập giữa các đặc trưng.

### 3. Support Vector Machine (SVM)

Tìm siêu phẳng tối ưu để phân tách email spam và ham.

## Chỉ số đánh giá

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

# Practice 2: Sales Amount Prediction

## Mục tiêu

Dự đoán giá trị doanh số (Sale Amount) dựa trên dữ liệu bán lẻ.

## Dataset

Dataset bao gồm thông tin về:

- Sản phẩm
- Số lượng bán
- Giá bán
- Danh mục sản phẩm
- Các thuộc tính liên quan khác

Dữ liệu được chia:

- 80% Training Set
- 20% Testing Set

## Các mô hình sử dụng

### 1. Linear Regression

Mô hình hồi quy tuyến tính dùng để dự đoán doanh số dựa trên mối quan hệ tuyến tính giữa các biến.

### 2. Decision Tree Regressor

Mô hình cây quyết định cho bài toán hồi quy.

Ưu điểm:

- Dễ giải thích
- Học được quan hệ phi tuyến

### 3. Random Forest Regressor

Mô hình tổ hợp nhiều cây quyết định nhằm tăng độ chính xác và giảm hiện tượng overfitting.

Ưu điểm:

- Độ chính xác cao
- Tổng quát hóa tốt
- Ổn định hơn Decision Tree

## Chỉ số đánh giá

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

---

# Công nghệ sử dụng

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

# Kết quả

Mỗi mô hình đều được đánh giá trên tập kiểm thử và so sánh dựa trên các chỉ số phù hợp với từng bài toán:

- Classification Metrics cho bài toán Spam Detection
- Regression Metrics cho bài toán Sales Prediction
