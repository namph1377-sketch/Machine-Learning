# Hướng dẫn Làm sạch & Tiền xử lý Dữ liệu Bán lẻ (Retail Sales Data)

Tài liệu này giải thích quy trình làm sạch, khám phá và chuẩn bị dữ liệu từ file `Dataset.ipynb`. Mục tiêu của file code là xử lý bộ dữ liệu `retail_sales_dataset.csv`, tạo đặc trưng mới và chuẩn bị dữ liệu để huấn luyện mô hình dự đoán `sales_amount`. Target cuối cùng được sử dụng là `sales_amount_log`.

---

## Mục lục

1. [Import thư viện và đọc dữ liệu](#1-import-thư-viện-và-đọc-dữ-liệu)  
2. [Data Overview](#2-data-overview)  
3. [Missing Data Analysis và Duplicate](#3-missing-data-analysis-và-duplicate)  
4. [Loại bỏ cột ID](#4-loại-bỏ-cột-id)  
5. [Univariate Analysis](#5-univariate-analysis)  
6. [Bivariate / Multivariate Analysis](#6-bivariate--multivariate-analysis)  
7. [Outlier Detection](#7-outlier-detection)  
8. [Feature Engineering trước khi split](#8-feature-engineering-trước-khi-split)  
9. [Split train/test theo thời gian](#9-split-traintest-theo-thời-gian)  
10. [Rolling feature sau khi split](#10-rolling-feature-sau-khi-split)  
11. [Encoding categorical variables](#11-encoding-categorical-variables)  
12. [Scaling đúng cách](#12-scaling-đúng-cách)  
13. [Lưu dữ liệu sau xử lý](#13-lưu-dữ-liệu-sau-xử-lý)

---

## 1. Import thư viện và đọc dữ liệu

Code bắt đầu bằng việc import các thư viện cần thiết và đọc file dữ liệu bán lẻ.

* **Thư viện sử dụng:** `pandas`, `numpy`, `matplotlib.pyplot`, `StandardScaler`.
* **File dữ liệu:** `retail_sales_dataset.csv`.
* **Lệnh đọc dữ liệu:** dùng `pd.read_csv()` để đưa dữ liệu vào DataFrame.
* **Mục tiêu:** chuẩn bị dữ liệu cho bài toán dự đoán doanh số bán hàng.

**Tác dụng:** tạo DataFrame ban đầu để bắt đầu quá trình kiểm tra, phân tích và tiền xử lý dữ liệu.

---

## 2. Data Overview

Bước này dùng để xem tổng quan cấu trúc ban đầu của dữ liệu.

* **Kích thước dữ liệu:** `df.shape` cho thấy dữ liệu có **120.000 dòng** và **17 cột**.
* **Tên cột:** dùng `df.columns.tolist()` để xem toàn bộ tên biến trong dataset.
* **Kiểu dữ liệu:** dùng `df.info()` để kiểm tra cột nào là số, cột nào là chuỗi.
* **Thống kê mô tả:** dùng `df.describe(include='all')` để xem count, unique, mean, std, min, max.
* **Các nhóm thông tin chính:** giao dịch, khách hàng, sản phẩm, giá, giảm giá, phương thức thanh toán, kênh bán hàng và khu vực.

**Tác dụng:** giúp hiểu dữ liệu trước khi xử lý, đồng thời phát hiện sớm các kiểu dữ liệu chưa phù hợp.

---

## 3. Missing Data Analysis và Duplicate

Bước này kiểm tra dữ liệu thiếu và dòng bị trùng lặp.

* **Kiểm tra missing value:** dùng `df.isnull().sum()` và tính thêm tỷ lệ phần trăm missing.
* **Kết quả missing:** tất cả các cột đều có `missing_count = 0`.
* **Kiểm tra duplicate:** dùng `df.duplicated().sum()`.
* **Kết quả duplicate:** số dòng trùng lặp là `0`.

**Tác dụng:** xác nhận dữ liệu không có giá trị rỗng và không có dòng trùng, nên không cần điền missing value hoặc xóa duplicate.

---

## 4. Loại bỏ cột ID

Code loại bỏ các cột định danh không cần thiết cho mô hình dự đoán.

* **Các cột bị xóa:** `transaction_id`, `customer_id`, `product_id`.
* **Lý do:** các cột này chỉ là mã định danh, không mô tả trực tiếp hành vi mua hàng hay doanh số.
* **Kết quả sau khi xóa:** dữ liệu còn **120.000 dòng** và **14 cột**.

**Tác dụng:** giảm nhiễu cho mô hình, tránh việc model học sai theo mã ID thay vì học quy luật thật của dữ liệu.

---

## 5. Univariate Analysis

Bước này phân tích từng biến riêng lẻ để hiểu phân phối dữ liệu.

* **Các biến số được kiểm tra:** `quantity`, `unit_price`, `discount_pct`, `sales_amount`.
* **Thống kê nổi bật:**
  * `quantity` dao động từ **1 đến 5**.
  * `unit_price` dao động từ **7.73 đến 493.51**.
  * `discount_pct` dao động từ **0 đến 30**.
  * `sales_amount` dao động từ **5.41 đến 2467.55**.
* **Phân tích theo nhóm:** notebook xem doanh số trung bình theo `category` và `transaction_month`.

**Tác dụng:** giúp phát hiện biến có phân phối lệch, khoảng giá trị lớn hoặc khả năng có outlier.

---

## 6. Bivariate / Multivariate Analysis

Bước này xem mối quan hệ giữa nhiều biến với nhau và với target `sales_amount`.

* **Correlation matrix:** dùng `df[num_cols].corr()` để xem tương quan giữa các biến số.
* **Kết quả đáng chú ý:**
  * `quantity` có tương quan khá cao với `sales_amount`.
  * `unit_price` cũng có tương quan khá cao với `sales_amount`.
  * `discount_pct` có tương quan âm nhẹ với `sales_amount`.
* **Trực quan hóa:** dùng ma trận tương quan để nhìn nhanh mối liên hệ giữa các biến.

**Tác dụng:** giúp xác định biến nào có ảnh hưởng nhiều đến doanh số và hỗ trợ bước chọn/tạo đặc trưng.

---

## 7. Outlier Detection

Bước này phát hiện giá trị ngoại lai của biến mục tiêu `sales_amount`.

* **Phương pháp sử dụng:** IQR Method, Z-score và boxplot.
* **IQR Method:** tính `Q1`, `Q3`, `IQR = Q3 - Q1`, sau đó xác định lower bound và upper bound.
* **Kết quả IQR:** phát hiện **8.309 outlier**, chiếm khoảng **6.92%** dữ liệu.
* **Kết quả Z-score:** phát hiện **2.846 outlier**, chiếm khoảng **2.37%** dữ liệu.
* **Cách xử lý:** chỉ phát hiện, **không xóa outlier**.

**Tác dụng:** giúp nhận biết target có giá trị doanh số rất cao. Không xóa ngay vì doanh số cao có thể là giao dịch thật, không nhất thiết là lỗi dữ liệu.

---

## 8. Feature Engineering trước khi split

Bước này tạo thêm các đặc trưng mới trước khi chia train/test.

* **Log transform target:** tạo cột `sales_amount_log = np.log1p(sales_amount)`.
* **Xử lý thời gian:** chuyển `transaction_date` sang kiểu datetime.
* **Tạo feature từ ngày tháng:**
  * `transaction_year`
  * `transaction_month`
  * `transaction_day`
  * `transaction_dayofweek`
* **Mã hóa nhóm tuổi:** chuyển `customer_age_group` thành `customer_age_group_encoded`.
* **Kết quả kiểm tra:** số dòng bị lỗi mapping nhóm tuổi là `0`.

**Tác dụng:** log transform giúp giảm độ lệch của target, còn feature ngày tháng giúp mô hình học được xu hướng theo thời gian.

---

## 9. Split train/test theo thời gian

Dữ liệu được sắp xếp theo `transaction_date`, sau đó chia train/test theo trục thời gian.

* **Cách chia:** 80% dữ liệu cũ làm train, 20% dữ liệu mới làm test.
* **Train shape:** `(96000, 20)`.
* **Test shape:** `(24000, 20)`.
* **Khoảng thời gian train:** từ `2024-01-01` đến `2025-08-07`.
* **Khoảng thời gian test:** từ `2025-08-07` đến `2025-12-30`.
* **Không dùng random split:** vì dữ liệu giao dịch có yếu tố thời gian.

**Tác dụng:** tránh dùng dữ liệu tương lai để dự đoán quá khứ, phù hợp hơn với bài toán dự đoán doanh số thực tế.

---

## 10. Rolling feature sau khi split

Bước này tạo đặc trưng trung bình động sau khi đã chia train/test.

* **Feature được tạo:** `qty_roll_mean_30d`.
* **Ý nghĩa:** số lượng bán trung bình trong 30 ngày trước đó của từng sản phẩm.
* **Cách tính:** nhóm theo `product_name`, sắp xếp theo `transaction_date`, rồi tính rolling mean.
* **Điểm quan trọng:** rolling feature được tạo **sau khi split** để hạn chế data leakage.
* **Test data:** không dùng `quantity` của chính tập test để tính lịch sử.

**Tác dụng:** giúp mô hình biết xu hướng bán gần đây của từng sản phẩm, từ đó hỗ trợ dự đoán doanh số tốt hơn.

---

## 11. Encoding categorical variables

Bước này chuyển các biến phân loại dạng chữ sang dạng số.

* **Các cột không dùng trực tiếp được drop:** `sales_amount`, `transaction_date`, `quantity`, `customer_age_group`.
* **One-hot encoding:** dùng `pd.get_dummies()` cho các biến phân loại.
* **Các biến được mã hóa:** `customer_gender`, `customer_segment`, `product_name`, `category`, `brand`, `payment_method`, `sales_channel`, `region`.
* **Căn chỉnh cột test:** dùng `reindex` để tập test có cùng cấu trúc cột với tập train.
* **Kết quả sau encoding:**
  * Train data: `(96000, 80)`.
  * Test data: `(24000, 80)`.
* **Target:** `sales_amount_log` vẫn được giữ trong dataframe.

**Tác dụng:** biến dữ liệu chữ thành các cột 0/1 để mô hình Machine Learning có thể xử lý.

---

## 12. Scaling đúng cách

Bước này chuẩn hóa các feature số bằng `StandardScaler`.

* **Các cột được chuẩn hóa:**
  * `unit_price`
  * `discount_pct`
  * `customer_age_group_encoded`
  * `qty_roll_mean_30d`
  * `transaction_year`
  * `transaction_month`
  * `transaction_day`
  * `transaction_dayofweek`
* **Cách làm đúng:** scaler chỉ `fit` trên train và chỉ `transform` trên test.
* **Không scale target:** `sales_amount_log` là target nên không bị scale chung với feature.
* **Kết quả train sau scaling:** các feature số có mean xấp xỉ `0` và std xấp xỉ `1`.

**Tác dụng:** đưa các biến số về cùng thang đo, giúp các mô hình như Linear Regression, Ridge, Lasso, KNN hoặc SVR học ổn định hơn.

---

## 13. Lưu dữ liệu sau xử lý

Bước cuối cùng là lưu dữ liệu đã xử lý thành file CSV.

* **File train:** `retail_train_80.csv`.
* **File test:** `retail_test_20.csv`.
* **Final train shape:** `(96000, 80)`.
* **Final test shape:** `(24000, 80)`.
* **Target column:** `sales_amount_log`.
* **Không tách riêng X/y:** target vẫn nằm trong bảng train/test để thuận tiện cho bước train model sau.

**Tác dụng:** tạo dữ liệu sạch, đã được encoding và scaling, sẵn sàng dùng cho mô hình hồi quy dự đoán doanh số.

