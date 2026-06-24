# HƯỚNG DẪN LÀM SẠCH VÀ TIỀN XỬ LÝ DỮ LIỆU BÁN LẺ

## 1. Mục tiêu xử lý dữ liệu

Notebook `Dataset_fixed_keep_cells_scale.ipynb` được dùng để làm sạch và tiền xử lý bộ dữ liệu bán lẻ `retail_sales_dataset.csv`. Mục tiêu cuối cùng là tạo ra hai tập dữ liệu đã sẵn sàng cho bước huấn luyện mô hình dự đoán doanh số bán hàng.

Biến mục tiêu chính của bài toán là `sales_amount_log`, được tạo từ `sales_amount` bằng phép biến đổi log nhằm giảm độ lệch phân phối của doanh số. Sau khi xử lý, dữ liệu được chia thành hai file:

- `retail_train_80.csv`: tập huấn luyện, chiếm 80% dữ liệu theo thời gian.
- `retail_test_20.csv`: tập kiểm tra, chiếm 20% dữ liệu theo thời gian.

Quy trình xử lý được thiết kế theo hướng hạn chế data leakage, đặc biệt ở các bước tạo rolling feature, one-hot encoding và chuẩn hóa dữ liệu.

---

## 2. Pipeline tổng quát

Quy trình tiền xử lý dữ liệu được thực hiện theo thứ tự sau:

1. Import thư viện cần thiết.
2. Đọc dữ liệu từ file CSV.
3. Xem tổng quan dữ liệu.
4. Kiểm tra missing values và duplicate.
5. Loại bỏ các cột ID không cần thiết.
6. Phân tích đơn biến để hiểu phân phối dữ liệu.
7. Phân tích quan hệ giữa các biến.
8. Phát hiện outlier bằng boxplot, IQR và Z-score.
9. Tạo biến mục tiêu `sales_amount_log`.
10. Xử lý cột thời gian `transaction_date`.
11. Tạo đặc trưng ngày tháng.
12. Mã hóa nhóm tuổi khách hàng.
13. Chia train/test theo thời gian.
14. Tạo rolling feature `qty_roll_mean_30d` sau khi đã chia train/test.
15. One-hot encoding sau khi đã chia train/test.
16. Chuẩn hóa một số cột số bằng StandardScaler.
17. Lưu dữ liệu đã xử lý thành hai file train/test.

---

## 3. Import thư viện và đọc dữ liệu

Bước đầu tiên là import các thư viện cần thiết cho quá trình xử lý dữ liệu. Các thư viện chính gồm:

- `pandas`: dùng để đọc, xử lý và biến đổi dữ liệu dạng bảng.
- `numpy`: hỗ trợ xử lý số học và giá trị thiếu.
- `matplotlib` và `seaborn`: dùng để trực quan hóa dữ liệu.
- `StandardScaler`: dùng để chuẩn hóa một số đặc trưng số.

Dữ liệu được đọc từ file `retail_sales_dataset.csv`. Sau khi đọc, dữ liệu được đưa vào DataFrame để phục vụ các bước kiểm tra, phân tích và tiền xử lý tiếp theo.

---

## 4. Tổng quan dữ liệu

Sau khi đọc dữ liệu, notebook kiểm tra cấu trúc ban đầu của dataset. Các thông tin được kiểm tra gồm:

- Số dòng và số cột của dữ liệu.
- Danh sách tên cột.
- Kiểu dữ liệu của từng cột.
- Một số dòng đầu tiên để quan sát dữ liệu thực tế.
- Thống kê mô tả của các biến số và biến phân loại.

Bước này giúp hiểu dữ liệu ban đầu gồm những nhóm thông tin nào, chẳng hạn như thông tin giao dịch, khách hàng, sản phẩm, giá bán, giảm giá, kênh bán hàng và khu vực.

---

## 5. Kiểm tra missing values và duplicate

Notebook kiểm tra dữ liệu thiếu bằng cách đếm số lượng giá trị rỗng ở từng cột. Đồng thời, dữ liệu cũng được kiểm tra duplicate để xem có dòng nào bị lặp lại hoàn toàn hay không.

Kết quả cho thấy dữ liệu không có missing values và không có dòng trùng lặp. Vì vậy, notebook không cần thực hiện bước điền giá trị thiếu hoặc xóa dòng duplicate.

Bước này quan trọng vì dữ liệu thiếu hoặc trùng lặp có thể làm sai lệch kết quả phân tích và ảnh hưởng đến quá trình huấn luyện mô hình.

---

## 6. Loại bỏ các cột ID

Các cột định danh như mã giao dịch, mã khách hàng và mã sản phẩm được loại bỏ khỏi dữ liệu dùng cho mô hình. Những cột này chỉ đóng vai trò nhận diện bản ghi, không phản ánh trực tiếp quy luật doanh số.

Các cột ID được loại bỏ gồm:

- `transaction_id`
- `customer_id`
- `product_id`

Điểm quan trọng là không xóa `quantity` ở bước này. Cột `quantity` vẫn được giữ lại vì cần dùng để tạo đặc trưng rolling mean `qty_roll_mean_30d` ở các bước sau.

---

## 7. Phân tích đơn biến

Notebook thực hiện phân tích từng biến riêng lẻ để hiểu rõ phân phối của dữ liệu. Các biến quan trọng được xem xét gồm:

- `quantity`
- `unit_price`
- `discount_pct`
- `sales_amount`

Bước này giúp nhận diện khoảng giá trị, xu hướng phân phối và khả năng xuất hiện ngoại lai. Ngoài ra, notebook cũng quan sát doanh số theo một số nhóm như danh mục sản phẩm hoặc thời gian giao dịch để hiểu sơ bộ hành vi bán hàng.

---

## 8. Phân tích quan hệ giữa các biến

Sau khi phân tích từng biến, notebook xem xét mối quan hệ giữa các biến số với nhau và với biến doanh số. Ma trận tương quan được sử dụng để đánh giá mức độ liên hệ giữa các biến như `quantity`, `unit_price`, `discount_pct` và `sales_amount`.

Kết quả phân tích giúp xác định những biến có khả năng ảnh hưởng mạnh đến doanh số. Thông thường, `quantity` và `unit_price` có ảnh hưởng lớn đến `sales_amount`, trong khi `discount_pct` có thể có tác động theo hướng khác tùy vào dữ liệu thực tế.

---

## 9. Phát hiện outlier

Notebook sử dụng nhiều phương pháp để phát hiện outlier trong dữ liệu, đặc biệt là với biến doanh số.

Các phương pháp được sử dụng gồm:

- Boxplot để quan sát trực quan giá trị bất thường.
- IQR Method để xác định các điểm nằm ngoài khoảng hợp lý.
- Z-score để phát hiện các điểm lệch xa so với trung bình.

Trong bài toán bán lẻ, outlier không nhất thiết là lỗi dữ liệu. Một giao dịch có doanh số cao có thể là giao dịch thật, ví dụ khách hàng mua số lượng lớn hoặc sản phẩm có giá cao. Vì vậy, notebook chỉ phát hiện và phân tích outlier, không xóa trực tiếp các dòng này.

---

## 10. Tạo biến mục tiêu `sales_amount_log`

Biến `sales_amount` là doanh số gốc của giao dịch. Tuy nhiên, doanh số thường có phân phối lệch phải, tức là phần lớn giao dịch có giá trị vừa hoặc thấp, còn một số ít giao dịch có giá trị rất cao.

Để giảm độ lệch này, notebook tạo thêm biến `sales_amount_log`. Đây là biến mục tiêu cuối cùng dùng cho mô hình hồi quy dự đoán doanh số.

Việc dùng log transform giúp mô hình học ổn định hơn, giảm ảnh hưởng quá mạnh của các giao dịch có doanh số rất lớn.

---

## 11. Xử lý cột thời gian

Cột `transaction_date` được chuyển về kiểu dữ liệu thời gian để có thể sắp xếp dữ liệu và tạo các đặc trưng liên quan đến ngày tháng.

Sau khi chuyển đổi, notebook kiểm tra lại kiểu dữ liệu để đảm bảo `transaction_date` đã được nhận diện đúng là dữ liệu ngày tháng.

Đây là bước quan trọng vì bài toán có yếu tố thời gian, nên train/test không nên chia ngẫu nhiên mà cần chia theo thứ tự thời gian.

---

## 12. Tạo đặc trưng ngày tháng

Từ cột `transaction_date`, notebook tạo thêm các đặc trưng thời gian gồm:

- `transaction_year`: năm giao dịch.
- `transaction_month`: tháng giao dịch.
- `transaction_day`: ngày trong tháng.
- `transaction_dayofweek`: thứ trong tuần.

Các đặc trưng này giúp mô hình học được xu hướng theo thời gian, ví dụ doanh số thay đổi theo tháng, theo ngày trong tuần hoặc theo từng giai đoạn trong năm.

---

## 13. Mã hóa nhóm tuổi khách hàng

Cột `customer_age_group` là biến phân loại dạng nhóm tuổi. Notebook chuyển biến này thành dạng số thông qua cột `customer_age_group_encoded`.

Việc mã hóa nhóm tuổi giúp mô hình có thể sử dụng thông tin tuổi khách hàng trong quá trình học. Sau khi mã hóa, notebook kiểm tra lại để đảm bảo không có nhóm tuổi nào bị lỗi mapping.

---

## 14. Chia train/test theo thời gian

Dữ liệu được sắp xếp theo `transaction_date`, sau đó chia thành train và test theo tỷ lệ 80/20.

- 80% dữ liệu cũ hơn được dùng làm tập train.
- 20% dữ liệu mới hơn được dùng làm tập test.

Cách chia này phù hợp với bài toán dự đoán doanh số vì trong thực tế, mô hình sẽ dùng dữ liệu quá khứ để dự đoán dữ liệu tương lai. Việc chia theo thời gian giúp tránh tình trạng mô hình nhìn thấy thông tin tương lai trong quá trình huấn luyện.

---

## 15. Tạo rolling feature `qty_roll_mean_30d`

Sau khi đã chia train/test, notebook tạo đặc trưng `qty_roll_mean_30d`. Đây là số lượng bán trung bình trong 30 ngày trước đó của từng sản phẩm.

Điểm quan trọng là rolling feature được tạo sau khi đã split dữ liệu. Việc này giúp tránh data leakage, vì tập test không được sử dụng thông tin từ chính nó hoặc dữ liệu tương lai để tạo đặc trưng.

Với tập train, rolling mean được tính dựa trên lịch sử trước dòng hiện tại. Với tập test, rolling mean chỉ được tính dựa trên dữ liệu lịch sử từ train. Nếu một sản phẩm không có đủ lịch sử trong 30 ngày trước đó, notebook sẽ điền bằng trung bình số lượng bán của sản phẩm đó trong tập train. Nếu vẫn không có, notebook dùng trung bình toàn cục của `quantity` trong train.

Cách xử lý này đảm bảo đặc trưng rolling được tạo hợp lý và không làm rò rỉ thông tin từ test vào train.

---

## 16. One-hot encoding sau khi split

Sau khi tạo xong rolling feature, notebook tiến hành one-hot encoding các biến phân loại.

Các biến phân loại được mã hóa gồm:

- `customer_gender`
- `customer_segment`
- `product_name`
- `category`
- `brand`
- `payment_method`
- `sales_channel`
- `region`

One-hot encoding được thực hiện sau khi đã chia train/test. Tập train và test được mã hóa riêng, sau đó test được căn chỉnh lại theo đúng danh sách cột của train. Việc này đảm bảo hai tập dữ liệu có cùng số cột, cùng tên cột và cùng thứ tự cột.

Các cột không dùng trực tiếp cho mô hình như `transaction_date`, `quantity`, `sales_amount` và `customer_age_group` được loại bỏ trước khi lưu dữ liệu cuối cùng.

---

## 17. Chuẩn hóa dữ liệu bằng StandardScaler

Notebook sử dụng StandardScaler để chuẩn hóa một số cột số. Scaler được fit trên tập train và chỉ transform trên tập test. Cách làm này đúng quy trình vì test không được tham gia vào quá trình học tham số chuẩn hóa.

Trong phiên bản code hiện tại, các cột được chuẩn hóa gồm:

- `unit_price`
- `qty_roll_mean_30d`

Hai cột này được chuẩn hóa để đưa về cùng thang đo, giúp mô hình học ổn định hơn. Biến `sales_amount_log` được giữ làm target. Nếu mục tiêu là xử lý đúng chuẩn machine learning, target không nên scale chung với feature.

Tuy nhiên, nếu muốn tái tạo đúng một dataset mẫu cũ đã scale cả target, có thể thêm `sales_amount_log` vào danh sách cột chuẩn hóa. Cách này chỉ nên dùng khi cần làm giống file mẫu, không phải lựa chọn chuẩn nhất cho pipeline ML.

---

## 18. Lưu dữ liệu sau xử lý

Sau khi hoàn tất các bước tiền xử lý, notebook lưu hai file dữ liệu cuối cùng:

- `retail_train_80.csv`
- `retail_test_20.csv`

Hai file này có cùng cấu trúc cột sau one-hot encoding và reindex. Tập train có 96.000 dòng, tập test có 24.000 dòng. Số lượng cột sau xử lý là 80 cột.

Việc lưu riêng train/test giúp thuận tiện cho bước huấn luyện và đánh giá mô hình ở các notebook tiếp theo.

---

## 19. Kiểm tra dữ liệu đầu ra

Sau khi lưu file, cần kiểm tra lại các điểm sau:

- Train và test có cùng số cột.
- Train và test có cùng tên cột.
- Train và test có cùng thứ tự cột.
- Không còn missing value trong dữ liệu đầu ra.
- `qty_roll_mean_30d` đã được tạo thành công.
- Các cột one-hot chỉ chứa giá trị 0 hoặc 1.
- Các cột được chuẩn hóa có trung bình gần 0 và độ lệch chuẩn gần 1 trên tập train.

Nếu các điều kiện trên đều đạt, dữ liệu đã sẵn sàng để dùng cho bước huấn luyện mô hình.

---

## 20. Tóm tắt các điểm đã sửa cho đúng quy trình

So với phiên bản code ban đầu, pipeline hiện tại đã sửa các lỗi quan trọng sau:

1. Không xóa `quantity` quá sớm vì cần dùng để tạo rolling feature.
2. Tạo `qty_roll_mean_30d` sau khi đã chia train/test.
3. Test không dùng `quantity` của chính test để tính rolling mean.
4. One-hot encoding được thực hiện sau khi chia train/test.
5. Test được reindex theo train để đảm bảo cùng cấu trúc cột.
6. StandardScaler chỉ fit trên train và transform trên test.
7. Không scale nhầm target `sales_amount_log` nếu đi theo chuẩn ML.
8. File train/test cuối cùng có cùng số cột và sẵn sàng để train model.

---

## 21. Kết luận

Quy trình tiền xử lý dữ liệu đã được xây dựng theo pipeline rõ ràng và phù hợp với bài toán dự đoán doanh số bán hàng. Các bước quan trọng như xử lý thời gian, tạo rolling feature, mã hóa biến phân loại, chuẩn hóa dữ liệu và chia train/test đều được thực hiện theo thứ tự hợp lý.

Điểm quan trọng nhất của pipeline là hạn chế data leakage. Rolling feature được tạo sau split, one-hot encoding được căn chỉnh giữa train và test, và scaler chỉ học tham số từ tập train. Nhờ đó, dữ liệu đầu ra phản ánh đúng tình huống thực tế: mô hình chỉ được học từ dữ liệu quá khứ để dự đoán dữ liệu tương lai.
