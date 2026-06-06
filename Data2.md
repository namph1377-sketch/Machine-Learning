# Hướng dẫn Làm sạch & Tiền xử lý Dữ liệu Bán lẻ (Retail Sales Data)
Tài liệu này giải thích chi tiết quy trình làm sạch, khám phá và chuẩn bị dữ liệu (Data Preprocessing & EDA) từ file Jupyter Notebook được cung cấp. Mục tiêu của file code là xử lý một tập dữ liệu về giao dịch bán lẻ ('retail_sales_dataset.csv'), qua đó phất hiện lỗi, loại bỏ các biến không cần thiết, mã hóa dữ liệu phân loại, và trích xuất các đặc trưng mới (Feature Engineering) để sẵn sàng nạp vào mô hình Machine Learning.

---

## Mục lục
1. [Tổng quan và Phát hiện lỗi] (#-tổng-quan-và-phát-hiện-lỗi)
2. [Xử lý Dữ liệu Thiếu (NaN) & Outliers] (#-xử-lý-dữ-liệu-thiếu-nan--outliers)
3.  [Mã hóa Dữ liệu Phân loại (Encoding)] (#-mã-hóa-dữ-liệu-phân-loại-encoding)
4. [Trích xuất Đặc trưng (Feature Engineering)] (#-trích-xuất-đặc-trưng-feature-engineering)
5. [Chuẩn hóa Thang đo (Scaling)] (#-chuẩn-hóa-thang-đo-scaling)
6. [Tách Dữ liệu Train/Test] (#-tách-dữ-liệu-traintest)

---

## Tổng quan và Phát hiện lỗi
COde bắt đầu bằng việc đọc tập dự liệu gồm **120.000 dòng** giao dịch bán lẻ
*   **Thư viện sử dụng:** 'pandas', 'numpy', 'matplotlib.pyplot'
*   **Kiểm tra trùng lặp:** Dùng 'df1.duplicated().sum()' để đếm số vòng lặp (Kết quả: '0')
*   **Kiểm tra đặc tính biến (Unique values):** Dùng 'unique()' để xem các giá trị của cột 'customer_gender' (Other, Male, Female), 'customer_segment', 'payment_method', v.v...
*   **Thống kê mô tả:** Hàm 'df1.describe()' cho thấy 'quantity' dao động từ 1 đến 5, 'unit_price' dao động mạnh (7.73 đến 493.51)
*   **Lọc biến thừa:** Loại bỏ các cột ID không mang giá trị cho mô hình dự đoán: 'transaction_id', 'customer_id', 'product_id'. Tập dữ liệu còn **14 cột**

## Xử lý Dữ liệu Thiếu (NaN) & Outliers
*   **Giá trị thiếu (NaN): **Kiểm tra bằng 'df2.isnull().sum()'. Kết quả cho thấy dữ liệu sạch, không có giá trị rỗng
*   **Ngoại lai (Outliers) cho 'unit_price':**
    *   Sử dụng Boxplot của matplotlib để kiểm tra phân phối
    *   Tính toán ngưỡng ngoại lại bằng phương pháp **IQR**: IQR = Q3 - Q1. Giá trị giới hạn: '(-313.475, 794.485)'. Không có outlier nào rơi ra ngoài vùng này.
*   **Ngoại lai ho sales_amount':** Tương tự, dùng Boxplot và IQR để kiểm tra nhưng phát hiẹn giá trị lệch.
*   **Biến đổi Log (Log Transformation):** Thay vì xóa các outlier của 'sales_amount', code áp dụng hàm logarit 'np.log1p(df2['sales_amount'])' để giảm độ lệch (skewness) của dữ liệu, tạo ra cột mới 'sales_amount_log'. Xóa cột 'sales_amount' gốc sau khi biến đổi

## Mã hóa Dữ liệu Phân loại (Encoding)
Dữ liệu chứa nhiều cột phân loại dạng chữ. Để mô hình tính toán được, code thực hiện hai loại mã hóa: 
1.  **Ordinal Encoding (Mã hóa thứ bậc):**
    *   Cột 'customer_age_group' chứa các dải tuổi có tính thứ tự
    *   Code dùng mapping thủ công (Map) dể chuyển thành số: '{'18-24': 0, '25-34': 1, '35-44': 2, '45-54': 3, '55+': 4}' -> Lưu vào cột 'customer_age_group_encoded'
2.  **One-Hot Encoding (Mã hóa phân loại độc lập):**
    *   Các cột như 'customer_gender', 'customer_segment', 'product_name', 'category', 'brand', 'payment_method', 'sales_channel', 'region' được xử lý bằng 'pd.get_dummies()'
    *   Kết quả là số lượng cột tăng vọt lên **78 cột**

## Trích xuất Đặc trưng (Feature Engineering)
Code trích xuất các đặc trưng mới (Feature) từ thời gian:
1.  **Tách ngày tháng:** Từ cột 'transaction_date', tạo thêm 4 cột mới: 'transaction_year', 'transaction_month', 'transaction_day', 'transaction_dayofweek'
2.  **Tính Moving Average (Trung bình động):**
    *   Dữ liệu được sắp xếp (sort) theo 'product_name' và 'transaction_date'
    *   Dùng hàm 'rolling('30D')' để tính số lượng bán trung bình (mean của 'quantity') trong 30 ngày trước đó cho từng sản phẩm. Tạo cột mới 'qty_roll_mean_30d'

## Chuẩn hóa Thang đo (Scaling)
Để tránh việc các biến có giá trị lớn (như price) lấn át các biến có giá trị nhỏ trong mô hình học máy:
*   **Công cụ:** Dùng 'StandardScaler' của 'sklearn.preprocessing'
*   **Các cột được chuẩn hóa:** 'unit_price', 'sales_amount_log', 'qty_roll_mean_30d'
*   **Tác dụng:** Đưa các giá trị về phân phối chuẩn (trung bình mean ~ 0, độ lệch chuẩn std = 1)
*   Cuối cùng, cột 'quantity' gốc được drop đi

## Tách Dữ liệu Train/Test
Dữ liệu là dạng chuỗi thời gian (time-series), nên không được trộn ngẫu nhiên (random split) mà phải tách theo 'trục thời gian'
*   Sắp xếp lại toàn bộ bảng theo 'transaction_date'
*   Drop cột 'transaction_date' vì mô hình không nhận định dạng datetime
*   **Chia 80/20:**
    *   Tập **Train** (80%): 'retail_train_80.csv' (96.000 dòng)
    *   Tập **Test** (20%): 'retail_test_20.csv' (24.000 dòng)