# KC House Data - Giới thiệu & Luồng Tiền Xử Lý Dữ Liệu (Pre_housing)

**Bài toán:** Cho một tập dữ liệu về nhà ở với các đặc trưng như diện tích (square footage), số phòng ngủ và vị trí, hãy dự đoán giá nhà bằng cách sử dụng mô hình Perceptron đa tầng (Multilayer Perceptron - MLP).

**Mục tiêu:** Dự đoán giá nhà (`price`) bằng mô hình MLP, dựa trên diện tích, chất lượng, vị trí và các đặc trưng đã kỹ thuật hóa.

**Mục lục:**

1. [Giới thiệu tổng quan về tập dữ liệu](#1-giới-thiệu-tổng-quan-về-tập-dữ-liệu)
2. [Giải thích các cột trong dataset](#2-giải-thích-các-cột-trong-dataset)
3. [Luồng Tiền Xử Lý Dữ Liệu](#3-luồng-tiền-xử-lý-dữ-liệu-chi-tiết-step-by-step)
   - [Bước 1: Import thư viện cần thiết](#bước-1-import-thư-viện-cần-thiết)
   - [Bước 2: Load dữ liệu](#bước-2-load-dữ-liệu)
   - [Bước 3: Xử lý dữ liệu trùng lặp](#bước-3-xử-lý-dữ-liệu-trùng-lặp)
   - [Bước 4: Khám phá dữ liệu (EDA)](#bước-4-khám-phá-dữ-liệu-eda)
   - [Bước 5: Xử lý Missing Values](#bước-5-xử-lý-missing-values)
   - [Bước 6: Xử lý Outlier](#bước-6-xử-lý-outlier)
   - [Bước 7–7.1: Feature Engineering & Vị trí](#bước-7-feature-engineering-tạo-đặc-trưng-mới)
   - [Bước 8–8.1: Zipcode & Chọn Feature](#bước-8-encoding-cho-zipcode)
   - [Lưu ý quan trọng khi huấn luyện](#lưu-ý-quan-trọng-khi-huấn-luyện)
   - [Bước 9: Chia tập dữ liệu](#bước-9-chia-tập-dữ-liệu)
   - [Bước 10: Feature Scaling](#bước-10-feature-scaling)
   - [Bước 11: Kiểm tra cuối cùng & Lưu dữ liệu](#bước-11-kiểm-tra-cuối-cùng--lưu-dữ-liệu)
4. [Tổng quan dữ liệu sau xử lý](#tổng-quan-dữ-liệu-thu-được-sau-quá-trình-xử-lý)

---

## 1. Giới thiệu tổng quan về tập dữ liệu

**Tên dataset:** KC House Data (King County House Sales)  
**Nguồn:** [https://www.kaggle.com/datasets/kamakshisoni/kchousedatanan/data](https://www.kaggle.com/datasets/kamakshisoni/kchousedatanan/data)  
**Số mẫu:** 21,613  
**Số cột:** 21 (+ cột index `Unnamed: 0` khi đọc CSV → tổng 22 cột)   


**Luồng dữ liệu qua các bước:**

| Biến tạm | Mô tả |
|----------|--------|
| `df` | Dữ liệu gốc sau khi load |
| `df1` | Sau điền missing values |
| `df2_` | Sau xử lý outlier & log-transform |
| `df3` | Sau feature engineering |
| `df4` | Bản phục vụ phân tích correlation (đã drop một số cột) |
| `X`, `y` | Ma trận feature và vector target cuối cùng |

---

## 2. Giải thích các cột trong dataset

| Cột                  | Ý nghĩa                                                                 | Kiểu dữ liệu | Ghi chú |
|----------------------|-------------------------------------------------------------------------|--------------|---------|
| `id`                 | Mã định danh duy nhất của mỗi giao dịch                                | Integer      | Không dùng để train |
| `date`               | **Ngày bán nhà** (ngày giao dịch hoàn tất)                             | String       | Trích xuất `year_sold`, `month_sold` |
| `price`              | **Giá bán cuối cùng** của ngôi nhà (**Target**)                        | Float        | Giữ nguyên giá trị gốc làm `y` |
| `bedrooms`           | Số lượng phòng ngủ                                                      | Integer      | Capping [1, 6] |
| `bathrooms`          | Số lượng phòng tắm (có thể là số thập phân, ví dụ 2.5)                 | Float        | Capping [0.5, 4.5] |
| `sqft_living`        | **Diện tích sàn nhà** Tổng diện tích tất cả các tầng có thể sử dụng để ở (bao gồm tầng hầm nếu có) - square feet            | Integer      | Tạo thêm `sqft_living_log`, bỏ khỏi `X` |
| `sqft_lot`           | Diện tích đất tổng thể (bao gồm nhà, sân vườn,...)                              | Integer      | - |
| `floors`             | Số tầng của ngôi nhà                                                    | Float        | - |
| `waterfront`         | Nhà có mặt tiền nhìn ra nước (biển, hồ) hay không (0 = Không, 1 = Có)  | Binary       | Ảnh hưởng mạnh đến giá |
| `view`               | Điểm đánh giá view (0 đến 4)                                            | Integer      | 0 = kém, 4 = tuyệt vời |
| `condition`          | Tình trạng tổng thể của ngôi nhà (1 đến 5)                             | Integer      | 1 = kém, 5 = rất tốt |
| `grade`              | Đánh giá chất lượng xây dựng & thiết kế (1 đến 13)                     | Integer      | **Rất quan trọng** |
| `sqft_above`         | Diện tích phần trên mặt đất (không kể tầng hầm)                        | Integer      | - |
| `sqft_basement`      | Diện tích tầng hầm                                                      | Integer      | Dùng tạo `has_basement` |
| `yr_built`           | Năm xây dựng ngôi nhà                                                   | Integer      | Dùng tạo `age`, sau đó drop |
| `yr_renovated`       | Năm cải tạo lần cuối (0 = chưa từng cải tạo)                           | Integer      | Dùng tạo feature cải tạo, sau đó drop |
| `zipcode`            | Mã bưu chính (vùng khu vực)                                            | Integer      | Frequency encoding → `zipcode_freq` |
| `lat`                | Vĩ độ (latitude)                                                        | Float        | Tọa độ địa lý |
| `long`               | Kinh độ (longitude)                                                     | Float        | Tọa độ địa lý |
| `sqft_living15`      | Diện tích trung bình của 15 ngôi nhà láng giềng gần nhất               | Integer      | So sánh với khu vực |
| `sqft_lot15`         | Diện tích đất trung bình của 15 ngôi nhà láng giềng gần nhất           | Integer      | So sánh với khu vực |

---

## 3. Luồng Tiền Xử Lý Dữ Liệu (Chi tiết Step by Step)

### Bước 1: Import thư viện cần thiết

- `pandas`, `numpy`
- `sklearn.model_selection` (`train_test_split`)
- `sklearn.preprocessing` (`StandardScaler`)
- `matplotlib`, `seaborn`

### Bước 2: Load dữ liệu

- Đọc file `kc_house_data_NaN.csv`
- Kiểm tra `df.head()`, `df.shape` → **(21,613 × 22)**
- Kiểm tra `df.info()`, `df.describe()`

### Bước 3: Xử lý dữ liệu trùng lặp

- `df.duplicated().sum()` → **0 dòng trùng**
- Không cần xóa duplicate

### Bước 4: Khám phá dữ liệu (EDA)

- **Missing values:** chỉ `bedrooms` (13) và `bathrooms` (10)
- **Correlation heatmap** toàn bộ biến số (`annot=True`)
- **Top feature tương quan với `price`:** `grade`, `sqft_living`, `sqft_above`, `sqft_living15`, `bathrooms`...

### Bước 5: Xử lý Missing Values

- Copy sang `df1 = df.copy()`
- Điền median cho từng cột có NaN

- Kiểm tra lại: tổng missing = **0**

### Bước 6: Xử lý Outlier

**6.1. Visualize trước xử lý**

- Boxplot 4 cột: `price`, `sqft_living`, `bedrooms`, `bathrooms`

**6.2. Log-transform (tạo cột mới, không ghi đè)**

- Copy `df2_ = df1.copy()`
- Tạo 2 cột log riêng cho biến liên tục lệch phải: 'price_log' tạo từ `price` và 'sqft_living_log' tạo từ `sqft_living`

- Boxplot sau log: `price_log`, `sqft_living_log`

**6.3. Xử lý outlier biến đếm bằng IQR + Capping**

| Cột | Phương pháp | Ngưỡng sau xử lý |
|-----|-------------|------------------|
| `bedrooms` | IQR kiểm tra → `np.clip` | [1, 6] |
| `bathrooms` | IQR kiểm tra → `np.clip` | [0.5, 4.5] |

- In bảng tổng hợp outlier trước khi capping
- Boxplot `bedrooms`, `bathrooms` sau xử lý


### Bước 7: Feature Engineering (Tạo đặc trưng mới)

Copy `df3 = df2_.copy()`

**Xử lý cột `date`:** ép kiểu datetime

**Các feature được tạo trong notebook:**

| Feature mới              | Tạo từ cột gốc                         | Mục đích |
|--------------------------|----------------------------------------|----------|
| `year_sold`              | `date`                                 | Năm căn nhà được bán |
| `month_sold`             | `date`                                 | Tháng bán (theo mùa/thị trường) |
| `age`                    | `year_sold` − `yr_built`               | Tuổi nhà tại thời điểm bán; âm → ép về 0 |
| `is_renovated`           | `yr_renovated`                         | 1 = đã cải tạo, 0 = chưa |
| `years_since_renovated`  | `yr_renovated`, `year_sold`, `age`     | Nếu đã cải tạo: `year_sold − yr_renovated`; nếu chưa: dùng `age` |
| `has_basement`           | `sqft_basement`                        | 1 = có tầng hầm, 0 = không |
| `price_log`              | `price`                                | Phiên bản log của giá (dùng làm feature) |
| `sqft_living_log`        | `sqft_living`                          | Phiên bản log diện tích (dùng làm feature) |
| `distance_to_center`     | `lat`, `long`                          | Khoảng cách Euclid đến trung tâm Seattle |
| `zipcode_freq`           | `zipcode`                              | Tần suất xuất hiện của mã zip |

### Bước 7.1: Xử lý đặc trưng vị trí (lat, long)

- Giữ nguyên `lat`, `long`
- Tạo `distance_to_center` với tọa độ tham chiếu Seattle: seattle_lat = 47.6062 và seattle_long = -122.3321.

- Vẽ scatter map vị trí nhà + biểu đồ `distance_to_center` vs `price` (có đường hồi quy)

**Mục đích:** Nhà gần trung tâm Seattle thường có giá cao hơn; feature giúp mô hình nắm yếu tố vị trí địa lý.

### Bước 8: Encoding cho `zipcode`

Dùng Frequency Encoding tạo thêm cột `zipcode_freq` giúp mô hình học được:

- Những zipcode phổ biến (nhiều nhà bán) thường là khu vực phát triển, có nhu cầu cao.
- Những zipcode hiếm có thể là khu vực ít dân cư hoặc xa trung tâm.


### Bước 8.1: Phân tích Correlation & chọn Feature

**Bước 1 — Phân tích tương quan:** Tạo `df4` từ `df3` (bỏ các cột thừa/đã thay thế), vẽ heatmap và xếp hạng tương quan với `price_log`. Top feature: `grade` (0.70), `sqft_living_log` (0.67), `sqft_living15` (0.62), `sqft_above` (0.60), `bathrooms` (0.55) → **chất lượng xây dựng** và **diện tích** ảnh hưởng giá mạnh nhất.

**Bước 2 — Tách `X` và `y`:** `y` = cột `price` (giá gốc USD); `X` = `df3` sau khi loại **9 cột** dưới đây, còn **23 feature**:

| Cột loại bỏ | Lý do |
|-------------|--------|
| `id`, `Unnamed: 0` | Mã định danh / index |
| `date` | Đã có `year_sold`, `month_sold` |
| `zipcode` | Đã có `zipcode_freq` |
| `yr_built`, `yr_renovated` | Đã có `age`, `is_renovated`, `years_since_renovated` |
| `price` | Tách riêng làm `y` |
| `sqft_living` | Thay bằng `sqft_living_log` |
| `year_sold` | Chỉ 2014–2015, biến thiên thấp |

23 feature trong `X`: `bedrooms`, `bathrooms`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `grade`, `sqft_above`, `sqft_basement`, `lat`, `long`, `sqft_living15`, `sqft_lot15`, `price_log`, `sqft_living_log`, `month_sold`, `age`, `is_renovated`, `years_since_renovated`, `has_basement`, `distance_to_center`, `zipcode_freq`

### Lưu ý quan trọng khi huấn luyện

- Các file **X_train, X_val, X_test** chứa feature `price_log` (biến mục tiêu đã được log transform).
- Các file **y_train, y_val, y_test** chứa **price gốc** (chưa transform).

Khi huấn luyện xong, model sẽ dự đoán ra **price_log**. → Sau khi train model và đánh giá cần phải tiến hành **chuyển ngược price_log** về giá trị **price** thật.


### Bước 9: Chia tập dữ liệu

| Tập | Số mẫu | Tỷ lệ |
|-----|--------|-------|
| Train | 15,128 | 70% |
| Validation | 3,243 | 15% |
| Test | 3,242 | 15% |

### Bước 10: Feature Scaling

- Sử dụng `StandardScaler`
- Lưu bản raw (`X_train_raw`, ...) để so sánh trước/sau scale
- **Fit chỉ trên Train**, transform Val & Test → tránh data leakage
- Kết quả: mean ≈ 0, std ≈ 1 trên tập Train

### Bước 11: Kiểm tra cuối cùng & Lưu dữ liệu

**Kiểm tra:**

- `X_train`: (15,128 × 23), `y_train`: (15,128,)
- `X_val`: (3,243 × 23), `y_val`: (3,243,)
- `X_test`: (3,242 × 23), `y_test`: (3,242,)
- Missing values trên cả 6 tập: **0**

**Lưu file CSV**:

| File | Nội dung |
|------|----------|
| `X_train.csv`, `X_val.csv`, `X_test.csv` | Ma trận feature đã scale |
| `y_train.csv`, `y_val.csv`, `y_test.csv` | Vector target `price` |
| `train_housing.csv`, `val_housing.csv`, `test_housing.csv` | File gộp (X + y) |

---

## Tổng quan dữ liệu thu được sau quá trình xử lý

- Có **23 feature** đã chuẩn hóa + **1 target** `price` (giá gốc)
- Chia thành 3 tập train / validation / test theo tỷ lệ **70 : 15 : 15**
- Sẵn sàng huấn luyện mô hình MLP