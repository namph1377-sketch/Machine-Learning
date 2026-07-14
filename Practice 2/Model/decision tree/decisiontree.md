# Pipeline notebook Decision Tree Regressor

##  Tóm tắt pipeline 
```text
→ Import library 
→ Build Decision Tree from scratch
→ Tune max_features
→ Train final model
→ Tune max_depth
→ Evaluate with MAE / MSE / RMSE / R²
→ Plot Actual vs Predicted + Residuals
→ Export prediction_result.csv
```

## 1. Mục tiêu
xây dựng một mô hình **Decision Tree Regressor** từ đầu bằng Python để dự đoán **doanh thu bán hàng** (`sales_amount_log`), sau đó đánh giá mô hình trên tập train/test và xuất file dự đoán.

## 2. Thư viện sử dụng
- `pandas`: đọc và xử lý dữ liệu
- `numpy`: tính toán số học và đánh giá mô hình
- `matplotlib`: trực quan hóa kết quả

## 3. Xây dựng mô hình Decision Tree Regressor từ đầu

### 3.1. Cấu trúc dữ liệu nút
định nghĩa lớp `Node` để lưu:
- `feature`: đặc trưng dùng để chia
- `threshold`: ngưỡng chia
- `left`, `right`: hai nhánh con
- `value`: giá trị dự đoán tại nút lá

### 3.2. Lớp `DecisionTreeRegressor`
Mô hình được tự cài đặt theo hướng OOP, không dùng `sklearn`. Các tham số chính:
- `max_depth`: độ sâu tối đa của cây
- `min_samples_split`: số mẫu tối thiểu để tiếp tục chia
- `max_features`: số đặc trưng được xem xét ở mỗi nút
- `random_state`: cố định ngẫu nhiên khi chọn đặc trưng

### 3.3. Quy trình huấn luyện
Quá trình huấn luyện bắt đầu từ `fit()` và xây cây bằng `_grow_tree()`.

Tại mỗi nút:
1. Kiểm tra điều kiện dừng:
   - đạt `max_depth`
   - số mẫu nhỏ hơn `min_samples_split`
   - phương sai của `y` bằng 0
2. Nếu chưa dừng, chọn ngẫu nhiên một tập đặc trưng theo `max_features`
3. Dùng `_best_split()` để tìm cặp `(feature, threshold)` tốt nhất theo **Variance Reduction**
4. Chia dữ liệu thành nhánh trái/phải
5. Gọi đệ quy để tiếp tục mọc cây

### 3.4. Cách chọn phép chia
duyệt qua tất cả giá trị ngưỡng ứng viên của từng đặc trưng và tính:
- phương sai nút cha
- phương sai hai nút con
- mức giảm phương sai

Phép chia có **Variance Reduction** lớn nhất sẽ được chọn.

### 3.5. Dự đoán
Hàm `predict()` sử dụng `_traverse_tree()` để đi từ nút gốc đến nút lá và trả về giá trị trung bình tại nút lá.

## 4. Hàm đánh giá mô hình
dùng hàm `evaluate_model()` để tính:
- `MAE`
- `MSE`
- `RMSE`
- `R² Train`
- `R² Test`

Do biến mục tiêu được lưu theo dạng log (`sales_amount_log`), nên trước khi tính metric, chuyển ngược về doanh thu thực bằng `np.expm1()`.

## 5. Đọc dữ liệu
hai file:
- `retail_train_80.csv`
- `retail_test_20.csv`

Sau đó:
- tách `X_train`, `y_train`
- tách `X_test`, `y_test`

## 6. Pipeline thực nghiệm

### Bước 1: So sánh `max_features`
Đầu tiên cố định:
- `max_depth = 10`
- `min_samples_split = 2`
- `random_state = 42`

Sau đó thử các giá trị:
- `5`
- `10`
- `15`
- `20`
- `None`

Kết quả được lưu vào bảng gồm:
- `Max Features`
- `MAE`
- `RMSE`
- `R2 Train`
- `R2 Test`

Mục đích của bước này là tìm số lượng đặc trưng phù hợp nhất tại mỗi lần chia.

### Bước 2: Huấn luyện mô hình cuối với cấu hình đã chọn
Notebook dùng:
- `max_depth = 10`
- `max_features = 15`
- `min_samples_split = 2`

Đây là mô hình chính được dùng để đánh giá cuối cùng.

### Bước 3: Thử thêm các giá trị `max_depth`
Tiếp tục kiểm tra:
- `max_depth = 20`
- `max_depth = 30`

với `max_features = 15` để quan sát hiện tượng overfitting / giảm khả năng tổng quát hóa.

### Bước 4: Trực quan hóa kết quả
- biểu đồ **Actual vs Predicted**
- biểu đồ **Residual Plot**

Mục tiêu:
- kiểm tra mức độ bám sát giữa dự đoán và giá trị thực
- quan sát sai số còn lại có phân bố hợp lý hay không

### Bước 5: Xuất kết quả
Tạo file:
- `prediction_result.csv`

File này gồm:
- toàn bộ feature của tập test
- cột `Actual`
- cột `Predicted`




