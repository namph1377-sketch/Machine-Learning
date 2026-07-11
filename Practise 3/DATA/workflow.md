
#  Customer Personality Analysis - Data Preprocessing Pipeline


---

## Tổng Quan Workflow (Data Architecture)

Quy trình xử lý dữ liệu được thiết kế theo mô hình đường ống tuần tự (Sequential Pipeline) bao gồm 5 giai đoạn chính:

```
 Dữ liệu gốc (29 cột)
       │
       ▼
 [Giai đoạn 1: Làm sạch dữ liệu] ───► Loại bỏ cột thừa (ID, Z_...), Lọc Outliers tuổi tác (>100), lọc outliers
       │
       ▼
 [Giai đoạn 2: Feature Engineering] ──► Tạo biến mới: DaysCustomer, Age, children
       │
       ▼
 [Giai đoạn 3: Điền khuyết toán học] ─► Imputation cột Income theo nhóm (Age Decade + Education)
       │
       ▼
 [Giai đoạn 4: Mã hóa & Chuẩn hóa] ──► OrdinalEncoder (Học vấn), hôn nhân , StandardScaler
       │
       ▼
 Ma trận đặc trưng sẵn sàng (X_scaled)

```

---

##  Chi Tiết Các Bước Triển Khai (Implementation Details)
link dataset : https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis 

khám phá dữ liệu
Sau khi đọc dữ liệu, notebook kiểm tra cấu trúc ban đầu của dataset. Các thông tin được kiểm tra gồm:

Số dòng và số cột của dữ liệu.
Danh sách tên cột.
Kiểu dữ liệu của từng cột.
Một số dòng đầu tiên để quan sát dữ liệu thực tế.
Thống kê mô tả của các biến số và biến phân loại.

 kiểm tra dữ liệu thiếu bằng cách đếm số lượng giá trị rỗng ở từng cột. Đồng thời, dữ liệu cũng được kiểm tra duplicate để xem có dòng nào bị lặp lại hoàn toàn hay không.

### 1. Làm sạch dữ liệu & Loại bỏ nhiễu (Data Cleaning)

* **Loại bỏ đặc trưng không mang thông tin:** * Cột `ID` được loại bỏ vì chỉ mang tính định danh.
* Cột `Z_CostContact` và `Z_Revenue` bị xóa bỏ do là thuộc tính hằng số (zero variance), không đóng góp giá trị cho việc tính toán toán học.


* **Lọc bỏ điểm dị biệt (Outliers) thô:** * Lọc và loại bỏ các bản ghi có số tuổi `Age > 100` (dữ liệu năm sinh phi thực tế) trước khi thực hiện bước điền khuyết, tránh làm lệch (skew) giá trị trung vị của các phân nhóm.

### 2. Feature Engineering

Biến đổi các thuộc tính gốc thô thành các thuộc tính mới có giá trị biểu diễn cao hơn:

* **Thời gian gắn bó (`DaysCustomer`):** Chuyển đổi chuỗi ngày tháng `Dt_Customer` thành số ngày tính từ thời điểm đăng ký của khách hàng đến ngày tham chiếu lớn nhất trong tập dữ liệu.
* **Tuổi tác (`Age`):** Chuyển đổi năm sinh `Year_Birth` thành số tuổi thực tế tại thời điểm khảo sát ($2021$).
* **children:** gộp teen và kid lại->giảm chiều
### 3. Điền khuyết dữ liệu 

* **Bài toán:** Cột `Income` chứa một số giá trị bị khuyết (`NaN`).
* **Giải pháp học thuật:** Thay vì sử dụng giá trị trung bình/trung vị toàn cục (Global Median) một cách cảm tính, pipeline thực hiện gom nhóm dữ liệu theo **Thập kỷ tuổi** (`_age_band = (Age // 10) * 10`) kết hợp với **Trình độ học vấn** (`Education`).
* **Ý nghĩa:** Điền giá trị khuyết bằng trung vị của nhóm (`Group Median`) giúp phản ánh chính xác kinh tế của các cá nhân có cùng phân khúc lứa tuổi và trình độ học vấn trong xã hội.

### 4. Mã hóa và Chuẩn hóa thang đo (Encoding & Feature Scaling)

* **Mã hóa thứ tự (`Education`):** Áp dụng `OrdinalEncoder` chuyển đổi các phạm trù chữ sang số theo cấp bậc học vấn tăng dần:

$$\text{Basic (0)}< \text{Graduation (1)} < \text{2n Cycle (2)}  < \text{Master (3)} < \text{PhD (4)}$$


* **Nén nhị phân (`Marital_Status`):** Thu gọn các trạng thái hôn nhân nhiễu thành biến nhị phân đại diện cho cấu trúc gia đình: `1` (Có đôi/Có cặp - Married, Together) và `0` (Một mình - Single, Divorced, Widow, Alone,...).
* **Chuẩn hóa dữ liệu số (`StandardScaler`):** Đưa toàn bộ các thuộc tính số về cùng một thang đo có $\mu = 0$ và $\sigma = 1$ bằng công thức:

$$z = \frac{x - \mu}{\sigma}$$



## Kết Quả Đầu Ra Của Pipeline (Expected Output)

Sau khi thực thi xong file mã nguồn xử lý data, cấu trúc dữ liệu thu được bao gồm:

1. **`X_scaled`:** Ma trận kích thước $(2237, 26)$ đã được làm sạch, điền khuyết và chuẩn hóa hoàn toàn về phân phối chuẩn.

---