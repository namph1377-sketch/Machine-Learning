# BÁO CÁO TIỀN XỬ LÝ DỮ LIỆU BÁN LẺ THEO PIPELINE 7 BƯỚC

## 1. Mục tiêu của bài toán

Bộ dữ liệu được sử dụng trong bài là `retail_sales_dataset.csv`. Đây là bộ dữ liệu bán lẻ, chứa thông tin về giao dịch, khách hàng, sản phẩm, giá bán, giảm giá, kênh bán hàng, khu vực và thời gian giao dịch.

Mục tiêu của quá trình tiền xử lý là làm sạch dữ liệu, biến đổi dữ liệu thô thành dữ liệu có thể đưa vào mô hình Machine Learning, sau đó xuất ra hai tập dữ liệu cuối cùng:

- `retail_train_80.csv`: tập huấn luyện, chiếm 80% dữ liệu cũ hơn theo thời gian.
- `retail_test_20.csv`: tập kiểm tra, chiếm 20% dữ liệu mới hơn theo thời gian.

Bài toán ở đây là **bài toán học có giám sát dạng hồi quy**. Mục tiêu là dự đoán doanh số bán hàng của từng giao dịch. Biến mục tiêu cuối cùng là `sales_amount_log`, được tạo từ biến doanh số gốc `sales_amount` bằng phép biến đổi log.

Vì `sales_amount_log` là giá trị số liên tục nên bài toán thuộc nhóm **Regression**, không phải Classification.

---

## 2. Pipeline tổng quát

Quy trình tiền xử lý dữ liệu được rút gọn và trình bày theo 7 bước chính:

1. Đọc dữ liệu và thống kê dữ liệu ban đầu.
2. Khám phá dữ liệu và tiền xử lý cơ bản.
3. Kiểm tra và xử lí ngoại lai.
4. Xử lý biến mục tiêu.
5. Mã hóa dữ liệu phân loại.
6. Feature engineering.
7. Chuẩn hóa, chia train/test và xuất dữ liệu.

Pipeline này được xây dựng nhằm đảm bảo dữ liệu đầu ra sạch, đúng định dạng, hạn chế data leakage và sẵn sàng cho bước huấn luyện mô hình.

---

## 3. Bước 1: Đọc dữ liệu và thống kê dữ liệu ban đầu

Ở bước đầu tiên, dữ liệu được đọc từ file `retail_sales_dataset.csv` bằng thư viện `pandas`. Sau khi đọc dữ liệu, notebook tiến hành kiểm tra tổng quan để hiểu dữ liệu thô ban đầu.

Các nội dung được kiểm tra gồm:

- Số dòng và số cột của dataset.
- Danh sách tên cột.
- Kiểu dữ liệu của từng cột.
- Một số dòng đầu tiên của dữ liệu.
- Thống kê mô tả của các biến số.
- Thống kê mô tả của các biến phân loại.
- Kiểm tra dữ liệu có bị thiếu hay không.
- Kiểm tra dữ liệu có dòng trùng lặp hay không.

Bước này giúp xác định dữ liệu ban đầu có những nhóm thông tin nào. Trong bộ dữ liệu bán lẻ, các nhóm thông tin chính thường gồm thông tin giao dịch, khách hàng, sản phẩm, giá bán, giảm giá, doanh số, thời gian giao dịch, kênh bán hàng và khu vực.

Việc thống kê dữ liệu ban đầu rất quan trọng vì nó giúp người làm mô hình biết dữ liệu có đủ dùng hay không, có lỗi định dạng không và có cần xử lý missing value hoặc duplicate hay không.

---

## 4. Bước 2: Khám phá dữ liệu và tiền xử lý cơ bản

Sau khi nắm được cấu trúc dữ liệu, notebook tiếp tục khám phá dữ liệu và thực hiện các bước tiền xử lý cơ bản.

Trước hết, các cột ID không cần thiết được loại bỏ khỏi dữ liệu dùng cho mô hình. Các cột này chỉ có vai trò định danh từng bản ghi, khách hàng hoặc sản phẩm, không phản ánh trực tiếp quy luật doanh số.

Các cột ID được loại bỏ gồm:

- `transaction_id`
- `customer_id`
- `product_id`


Bên cạnh đó, notebook kiểm tra missing value bằng cách đếm số lượng giá trị thiếu ở từng cột. Nếu dữ liệu không có missing value thì không cần điền giá trị thiếu. Việc không tự ý fill dữ liệu khi không có missing value giúp tránh làm thay đổi dữ liệu gốc không cần thiết.

Notebook cũng kiểm tra duplicate để đảm bảo không có dòng bị lặp hoàn toàn. Nếu không có duplicate thì không cần xóa dòng nào.

Mục đích của bước này là loại bỏ thông tin không cần thiết, giữ lại các biến có ý nghĩa cho mô hình và đảm bảo dữ liệu cơ bản không bị lỗi trước khi xử lý sâu hơn.

---

## 5. Bước 3: Kiểm tra và xử lí ngoại lai

Sau bước tiền xử lý cơ bản, notebook kiểm tra ngoại lai, đặc biệt tập trung vào biến doanh số `sales_amount`.

Trong dữ liệu bán lẻ, doanh số có thể bị lệch phải. Điều này có nghĩa là phần lớn giao dịch có giá trị vừa hoặc thấp, nhưng vẫn có một số giao dịch có doanh số rất cao. Những điểm doanh số cao này có thể là ngoại lai, nhưng không nhất thiết là dữ liệu sai.

Các phương pháp được sử dụng để kiểm tra ngoại lai gồm:

- Histogram: dùng để quan sát phân phối của biến `sales_amount`.
- Boxplot: dùng để phát hiện trực quan các điểm nằm xa phần lớn dữ liệu.
- IQR Method: dùng để xác định các giá trị nằm ngoài khoảng hợp lý theo tứ phân vị.

Công thức IQR thường được hiểu như sau:

```text
IQR = Q3 - Q1
Lower Bound = Q1 - 1.5 * IQR
Upper Bound = Q3 + 1.5 * IQR
```

Các điểm nhỏ hơn `Lower Bound` hoặc lớn hơn `Upper Bound` có thể được xem là ngoại lai.

Tuy nhiên, trong bài toán bán lẻ, các giao dịch có doanh số cao có thể là giao dịch thật, ví dụ khách hàng mua số lượng lớn hoặc mua sản phẩm có giá trị cao. Vì vậy, pipeline chỉ phát hiện và phân tích ngoại lai, không xóa trực tiếp các dòng này.

Việc giữ lại các giao dịch doanh số cao giúp mô hình vẫn học được những trường hợp thực tế quan trọng trong hoạt động bán hàng.

---

## 6. Bước 4: Xử lý biến mục tiêu

Biến mục tiêu gốc của bài toán là `sales_amount`, thể hiện doanh số của từng giao dịch. Tuy nhiên, `sales_amount` thường có phân phối lệch phải, làm cho mô hình dễ bị ảnh hưởng mạnh bởi các giao dịch có doanh số rất lớn.

Để giảm độ lệch của phân phối, notebook tạo thêm biến:

```text
sales_amount_log
```

Biến này được tạo bằng phép biến đổi log từ `sales_amount`. Mục đích của log transform là làm cho phân phối doanh số bớt lệch hơn, giúp mô hình học ổn định hơn và giảm ảnh hưởng quá mạnh của các giá trị doanh số lớn.

Sau bước này, `sales_amount_log` được xem là biến mục tiêu chính của bài toán.

Nói cách khác:

```text
Input X: các đặc trưng sau tiền xử lý
Output y: sales_amount_log
```

Điểm quan trọng là `sales_amount` không được đưa vào tập đặc trưng đầu vào cuối cùng, vì nó là biến gốc dùng để tạo target. Nếu đưa `sales_amount` vào input, mô hình có thể bị rò rỉ đáp án.

---

## 7. Bước 5: Mã hóa dữ liệu phân loại

Trong dữ liệu ban đầu có nhiều cột dạng chữ. Các mô hình Machine Learning không thể học trực tiếp từ dữ liệu dạng text, vì vậy cần chuyển các biến phân loại thành dạng số.

Pipeline sử dụng hai cách mã hóa chính:

### 7.1. Ordinal Encoding cho nhóm tuổi

Cột `customer_age_group` là biến phân loại có thứ tự, ví dụ các nhóm tuổi từ trẻ đến lớn tuổi. Vì các nhóm này có quan hệ thứ bậc nên notebook mã hóa thành cột số:

```text
customer_age_group_encoded
```

Việc mã hóa này giúp mô hình hiểu được thông tin nhóm tuổi khách hàng ở dạng số.

### 7.2. One-hot Encoding cho các biến phân loại còn lại

Các cột phân loại không có thứ tự rõ ràng được mã hóa bằng one-hot encoding. Các cột này gồm:

- `customer_gender`
- `customer_segment`
- `product_name`
- `category`
- `brand`
- `payment_method`
- `sales_channel`
- `region`

One-hot encoding sẽ biến mỗi giá trị phân loại thành một cột nhị phân 0/1. Ví dụ, nếu cột `sales_channel` có hai giá trị `Online` và `Offline`, sau khi one-hot encoding có thể tạo thành các cột như `sales_channel_Online` và `sales_channel_Offline`.

Trong pipeline chuẩn, one-hot encoding cần được thực hiện cẩn thận để train và test có cùng cấu trúc cột. Vì vậy, sau khi mã hóa, tập test được căn chỉnh lại theo danh sách cột của tập train.

---

## 8. Bước 6: Feature engineering

Feature engineering là bước tạo thêm các đặc trưng mới có ý nghĩa hơn từ dữ liệu ban đầu. Mục tiêu là giúp mô hình có thêm thông tin để học tốt hơn.

### 8.1. Xử lý cột thời gian

Cột `transaction_date` được chuyển sang kiểu dữ liệu thời gian. Sau đó, dữ liệu được sắp xếp theo thời gian để phục vụ việc chia train/test và tạo đặc trưng lịch sử.

Từ `transaction_date`, notebook tạo thêm các đặc trưng ngày tháng gồm:

- `transaction_year`: năm giao dịch.
- `transaction_month`: tháng giao dịch.
- `transaction_day`: ngày trong tháng.
- `transaction_dayofweek`: thứ trong tuần.

Các đặc trưng này giúp mô hình học được xu hướng doanh số theo thời gian, ví dụ doanh số có thể thay đổi theo tháng, theo ngày trong tuần hoặc theo từng giai đoạn trong năm.

### 8.2. Tạo rolling feature

Một đặc trưng quan trọng được tạo thêm là:

```text
qty_roll_mean_30d
```

Đây là số lượng bán trung bình trong 30 ngày trước đó của từng sản phẩm. Đặc trưng này giúp mô hình biết được xu hướng bán gần đây của sản phẩm.

Điểm quan trọng là rolling feature cần được tạo theo hướng tránh data leakage. Tập test không được dùng thông tin của chính nó hoặc thông tin tương lai để tạo rolling mean.

Với tập train, rolling mean được tính dựa trên lịch sử trước dòng hiện tại. Với tập test, rolling mean chỉ được tính dựa trên dữ liệu lịch sử từ tập train. Nếu sản phẩm không có đủ lịch sử trong 30 ngày trước đó, giá trị thiếu được thay bằng trung bình `quantity` của sản phẩm trong train. Nếu vẫn không có, dùng trung bình toàn cục của `quantity` trong train.

Sau khi tạo xong `qty_roll_mean_30d`, cột `quantity` sẽ không được giữ trực tiếp trong dữ liệu cuối cùng vì thông tin lịch sử cần thiết đã được chuyển thành rolling feature.

---

## 9. Bước 7: Chuẩn hóa, chia train/test và xuất dữ liệu

Bước cuối cùng là chuẩn hóa dữ liệu, chia train/test và xuất dữ liệu đã xử lý thành file CSV.

### 9.1. Chia train/test theo thời gian

Dữ liệu được sắp xếp theo `transaction_date`, sau đó chia theo tỷ lệ 80/20:

- 80% dữ liệu cũ hơn dùng làm tập train.
- 20% dữ liệu mới hơn dùng làm tập test.

Cách chia này phù hợp với bài toán dự đoán doanh số vì trong thực tế, mô hình sẽ dùng dữ liệu quá khứ để dự đoán dữ liệu tương lai. Không nên chia ngẫu nhiên vì có thể làm mô hình nhìn thấy thông tin tương lai trong quá trình huấn luyện.

### 9.2. Chuẩn hóa dữ liệu số

Notebook sử dụng `StandardScaler` để chuẩn hóa một số cột số về cùng thang đo. Các cột được chuẩn hóa gồm:

- `unit_price`
- `qty_roll_mean_30d`

`StandardScaler` được fit trên tập train, sau đó dùng tham số đã học từ train để transform tập test. Đây là cách làm đúng vì tập test không được tham gia vào quá trình học tham số chuẩn hóa.

Biến mục tiêu `sales_amount_log` không nên scale chung với feature nếu làm đúng pipeline Machine Learning. Target được giữ riêng để phục vụ huấn luyện và đánh giá mô hình hồi quy.

### 9.3. Loại bỏ các cột không dùng trực tiếp cho mô hình

Trước khi xuất dữ liệu cuối cùng, các cột không dùng trực tiếp cho mô hình được loại bỏ, gồm:

- `transaction_date`
- `quantity`
- `sales_amount`
- `customer_age_group`

Trong đó, `quantity` chỉ bị loại bỏ sau khi đã dùng để tạo `qty_roll_mean_30d`.

### 9.4. Xuất dữ liệu

Sau khi hoàn tất tiền xử lý, dữ liệu được xuất thành hai file:

- `retail_train_80.csv`
- `retail_test_20.csv`

Hai file này có cùng số cột, cùng tên cột và cùng thứ tự cột. Điều này đảm bảo dữ liệu đã sẵn sàng cho bước huấn luyện và kiểm tra mô hình.

---

## 10. Kiểm tra dữ liệu đầu ra

Sau khi xuất file, cần kiểm tra lại các điều kiện sau:

- Train và test có cùng số cột.
- Train và test có cùng tên cột.
- Train và test có cùng thứ tự cột.
- Không còn missing value trong dữ liệu đầu ra.
- Cột `qty_roll_mean_30d` đã được tạo thành công.
- Các cột one-hot chỉ chứa giá trị 0 hoặc 1.
- Các cột được chuẩn hóa có trung bình gần 0 và độ lệch chuẩn gần 1 trên tập train.
- Biến mục tiêu cuối cùng là `sales_amount_log`.

Nếu các điều kiện trên đều đạt, dữ liệu đã sẵn sàng để đưa vào mô hình Machine Learning.

---

## 11. Kết luận

Pipeline tiền xử lý dữ liệu bán lẻ được xây dựng theo 7 bước rõ ràng: đọc và thống kê dữ liệu, khám phá và xử lý cơ bản, kiểm tra ngoại lai, xử lý biến mục tiêu, mã hóa dữ liệu phân loại, tạo đặc trưng mới, cuối cùng là chuẩn hóa, chia train/test và xuất dữ liệu.

Điểm quan trọng nhất của pipeline là hạn chế data leakage. Dữ liệu được chia theo thời gian để mô phỏng đúng tình huống thực tế, trong đó mô hình chỉ được học từ dữ liệu quá khứ để dự đoán dữ liệu tương lai. Rolling feature được tạo cẩn thận để test không dùng thông tin của chính test, one-hot encoding được căn chỉnh giữa train và test, và StandardScaler chỉ fit trên train.

Sau pipeline này, dữ liệu đầu ra có thể được sử dụng cho bài toán hồi quy dự đoán doanh số bán hàng với biến mục tiêu là `sales_amount_log`.
