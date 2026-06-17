# Mục Lục (Table of Contents)

1. [Linear Regression](#linear-regression)
   - [Loss](#loss)
     - [Type of loss](#type-of-loss)
     - [Cách chọn Loss](#cách-chọn-loss)
   - [Gradient Descent](#gradient-descent)
     - [Công thức](#công-thức)
       - [1. Hàm dự đoán (Prediction Function)](#1-hàm-dự-đoán-prediction-function)
       - [2. Hàm mất mát - MSE (Mean Squared Error)](#2-hàm-mất-mát---mse-mean-squared-error)
       - [3. Gradient (Đạo hàm) của hàm MSE](#3-gradient-đạo-hàm-của-hàm-mse)
       - [4. Cập nhật tham số (Parameter Update)](#4-cập-nhật-tham-số-parameter-update)
       - [5. Các khái niệm quan trọng](#5-các-khái-niệm-quan-trọng)
       - [6. Tóm tắt các công thức chính](#6-tóm-tắt-các-công-thức-chính)
     - [Model Convergence và Loss Curves](#model-convergence-và-loss-curves-đường-cong-hội-tụ-và-đường-cong-loss)
     - [Hàm lồi (Convex Functions) và "Ngọn núi một đỉnh"](#hàm-lồi-convex-functions-và-ngọn-núi-một-đỉnh)
     - [Hyperparameters (Siêu tham số)](#hyperparameters-siêu-tham-số)
       - [Learning rate (Tốc độ học)](#learning-rate-tốc-độ-học)
       - [Batch size (Kích thước lô)](#batch-size-kích-thước-lô)
       - [Epoch (Kỷ nguyên)](#epoch-kỷ-nguyên)

2. [Logistic Regression](#logistic-regression)
   - [Công thức hàm Sigmoid](#công-thức-hàm-sigmoid)
   - [Các đặc điểm quan trọng của hàm Sigmoid](#các-đặc-điểm-quan-trọng-của-hàm-sigmoid)
   - [Cách chuyển từ Xác suất sang Phân loại (Classification)](#cách-chuyển-từ-xác-suất-sang-phân-loại-classification)
   - [Tại sao cần Sigmoid trong ML?](#tại-sao-cần-sigmoid-trong-ml)
   - [Tóm tắt công thức tổng quát](#tóm-tắt-công-thức-tổng-quát)
   - [Hàm mất mát (Loss function) và Điều quy hóa (Regularization)](#hàm-mất-mát-loss-function-và-điều-quy-hóa-regularization)
     - [Hàm mất mát: Log Loss](#hàm-mất-mát-log-loss)
     - [Công thức Log Loss](#công-thức-log-loss)
     - [Sự phức tạp của mô hình và Overfitting](#sự-phức-tạp-của-mô-hình-và-overfitting)
     - [Điều quy hóa L2 (L2 Regularization)](#điều-quy-hóa-l2-l2-regularization)
     - [Tác động của siêu tham số Lambda (λ)](#tác-động-của-siêu-tham-số-lambda-λ)

3. [Classification](#classification)
   - [Threshold (Ngưỡng phân loại)](#threshold--ngưỡng-phân-loại-)
     - [Quy tắc ánh xạ](#quy-tắc-ánh-xạ)
     - [Ngưỡng 0.5 có phải luôn là tốt nhất?](#ngưỡng-05-có-phải-luôn-là-tốt-nhất)
     - [Tác động của việc thay đổi ngưỡng](#tác-động-của-việc-thay-đổi-ngưỡng)
   - [Confusion Matrix (Ma trận nhầm lẫn)](#confusion-matrix-ma-trận-nhầm-lẫn)
     - [Cấu trúc ma trận](#cấu-trúc-ma-trận)
     - [Giải thích 4 thành phần chính](#giải-thích-4-thành-phần-chính-rất-quan-trọng)
     - [Tại sao cần Confusion Matrix?](#tại-sao-cần-confusion-matrix)
     - [Các chỉ số rút ra từ Ma trận nhầm lẫn](#các-chỉ-số-rút-ra-từ-ma-trận-nhầm-lẫn)
     - [Lựa chọn chỉ số và Sự đánh đổi](#lựa-chọn-chỉ-số-và-sự-đánh-đổi)
   - [Đường cong ROC và Chỉ số AUC](#đường-cong-roc-và-chỉ-số-auc)
     - [Đường cong ROC (Receiver Operating Characteristic)](#đường-cong-roc-receiver-operating-characteristic)
     - [AUC (Area Under the ROC Curve)](#auc-area-under-the-roc-curve)
     - [Tại sao AUC lại quan trọng?](#tại-sao-auc-lại-quan-trọng)
     - [Những lưu ý (Caveats) khi dùng AUC](#những-lưu-ý-caveats-khi-dùng-auc)
   - [Độ lệch dự đoán (Prediction Bias)](#độ-lệch-dự-đoán-prediction-bias)
     - [Định nghĩa](#định-nghĩa)
     - [Ý nghĩa của các giá trị Bias](#ý-nghĩa-của-các-giá-trị-bias)
     - [Cách chẩn đoán: Chia nhóm (Bucketing)](#cách-chẩn-đoán-chia-nhóm-bucketing)
     - [Nguyên nhân gây ra Prediction Bias](#nguyên-nhân-gây-ra-prediction-bias)
     - [Cảnh báo quan trọng: Đừng "sửa ngọn"](#cảnh-báo-quan-trọng-đừng-sửa-ngọn)
   - [Phân loại đa lớp (Multi-class Classification)](#phân-loại-đa-lớp-multi-class-classification)
     - [Hai phương pháp tiếp cận chính](#hai-phương-pháp-tiếp-cận-chính)
     - [Phân biệt các loại Softmax](#phân-biệt-các-loại-softmax)
     - [Single-label vs. Multi-label (Quan trọng)](#single-label-vs-multi-label-quan-trọng)

4. [Vector đặc trưng (Feature Vectors)](#vector-đặc-trưng-feature-vectors)
   - [Định nghĩa](#định-nghĩa-1)
   - [Tại sao cần Vector đặc trưng?](#tại-sao-cần-vector-đặc-trưng)
   - [Các quy tắc thiết kế Vector đặc trưng tốt](#các-quy-tắc-thiết-kế-vector-đặc-trưng-tốt)
   - [Các bước biến đổi dữ liệu (Feature Transformation)](#các-bước-biến-đổi-dữ-liệu-feature-transformation)

5. [Chuẩn hóa dữ liệu (Normalization)](#chuẩn-hóa-dữ-liệu-normalization)
   - [Tại sao cần chuẩn hóa?](#tại-sao-cần-chuẩn-hóa)
   - [Các kỹ thuật chuẩn hóa phổ biến](#các-kỹ-thuật-chuẩn-hóa-phổ-biến)
     - [a. Linear Scaling (Thang đo tuyến tính)](#a-linear-scaling-thang-đo-tuyến-tính)
     - [b. Z-score (Standardization - Chuẩn hóa điểm Z)](#b-z-score-standardization---chuẩn-hóa-điểm-z)
     - [c. Log Scaling (Thang đo Logarit)](#c-log-scaling-thang-đo-logarit)
     - [d. Clipping (Cắt cụt)](#d-clipping-cắt-cụt)
   - [Khi nào thì dùng kỹ thuật nào?](#khi-nào-thì-dùng-kỹ-thuật-nào)

6. [Binning (Bucketing) - Chia nhóm dữ liệu](#binning-bucketing---chia-nhóm-dữ-liệu)
   - [Định nghĩa](#định-nghĩa-2)
   - [Tại sao cần Binning?](#tại-sao-cần-binning)
   - [Cách thực hiện](#cách-thực-hiện)
   - [Các chiến lược chia thùng phổ biến](#các-chiến-lược-chia-thùng-phổ-biến)

7. [Data Scrubbing (Làm sạch dữ liệu)](#data-scrubbing-làm-sạch-dữ-liệu)
   - [Tầm quan trọng](#tầm-quan-trọng)
   - [Xử lý dữ liệu thiếu (Missing Values)](#xử-lý-dữ-liệu-thiếu-missing-values)
   - [Xử lý giá trị ngoại lai (Outliers)](#xử-lý-giá-trị-ngoại-lai-outliers)
   - [Kiểm tra tính nhất quán (Consistency)](#kiểm-tra-tính-nhất-quán-consistency)
   - [Dữ liệu không đáng tin cậy](#dữ-liệu-không-đáng-tin-cậy)

8. [Biến đổi đa thức (Polynomial Transforms)](#biến-đổi-đa-thức-polynomial-transforms)
   - [Định nghĩa](#định-nghĩa-3)
   - [Tại sao cần Biến đổi đa thức?](#tại-sao-cần-biến-đổi-đa-thức)
   - [Ví dụ về công thức](#ví-dụ-về-công-thức)
   - [Đặc điểm quan trọng](#đặc-điểm-quan-trọng)
   - [Nguy cơ Overfitting (Quá khớp)](#nguy-cơ-overfitting-quá-khớp)

# Linear regression

```

y` = b + w1x1 + w2x2 + w3x3 + w4x4 + w5x5

```
 Trong đó 
 + y': predicted label
 + b: bias -> tính toán từ traing
 + w: weight -> tính toán từ traing
 + x: feature value
---
## Loss

Mô tả mức độ lệch của prediction. Đo lường khoảng cách giữa các dự đoán của mô hình và label thực tế

### Type of loss

| Loại loss | Định nghĩa | Phương trình |
|-----------|------------|----------------|
| L₁ loss | Tổng các giá trị tuyệt đối của chênh lệch giữa giá trị thực tế và giá trị dự đoán. | $\sum \|\text{giá trị thực} - \text{giá trị dự đoán}\|$ |
| Mean absolute error (MAE) | Trung bình của các L₁ loss trên một tập gồm N mẫu. | $\frac{1}{N} \sum \|\text{giá trị thực} - \text{giá trị dự đoán}\|$ |
| L₂ loss | Tổng bình phương chênh lệch giữa giá trị thực tế và giá trị dự đoán. | $\sum (\text{giá trị thực} - \text{giá trị dự đoán})^2$ |
| Mean squared error (MSE) | Trung bình của các L₂ loss trên một tập gồm N mẫu. | $\frac{1}{N} \sum (\text{giá trị thực} - \text{giá trị dự đoán})^2$ |
| Root mean squared error (RMSE) | Căn bậc hai của mean squared error (MSE). | $\sqrt{\frac{1}{N} \sum (\text{giá trị thực} - \text{giá trị dự đoán})^2}$ |

--> MAE và RMSE thường được dùng hơn so với L2 loss hoặc MSE trong một số trường hợp sử dụng vì chúng dễ hiểu hơn đối với con người, do chúng đo lường lỗi bằng cùng thang đo với giá trị dự đoán của mô hình.

---

### Cách chọn Loss 

Khi chọn loss, cân nhắc cách bạn muốn mô hình xử lý các giá trị ngoại lệ.
MSE làm cho mô hình dịch chuyển nhiều hơn về phía các giá trị ngoại lệ, trong khi MAE thì không. L2 gây ra mức phạt cao hơn nhiều đối với một giá trị ngoại lệ so với mất mát L1.

MAE model:

![MAE model](images/model-mae.png "MAE")

MSE model:

![MSE model](images/model-mse.png "MSE")

---
## Gradient descent

Kỹ thuật toán học tìm kiếm lặp đi lặp lại các trọng số và độ lệch sao cho tạo ra mô hình có loss thấp nhất. Gradient descent tìm ra trọng số và độ lệch tốt nhất bằng cách lặp lại quy trình sau đây với số lần lặp do người dùng xác định.

+ B1: Tính toán loss với weight hiện tại và bias
+ B2: Xác định hướng di chuyển của các trọng số và độ lệch để giảm thiểu loss
+ B3: Điều chỉnh giá trị trọng số và độ lệch một chút theo hướng giảm loss
+ B4: Quay lại bước một và lặp lại quy trình cho đến khi mô hình không thể giảm thiểu loss thêm nữa

![gradient descent](images/gradient-descent.png "gradient descent")

---

### Công thức

#### 1. Hàm dự đoán (Prediction Function)

Với một đặc trưng (feature) `x`, trọng số `w`, và độ chệch `b`:

$$ f_{w,b}(x) = (w \times x) + b $$

Trong đó:
- `x`: giá trị đặc trưng đầu vào (ví dụ: trọng lượng xe - nghìn pounds)
- `w`: trọng số (weight) - hệ số góc
- `b`: độ chệch (bias) - điểm chắn trục tung
- `f_{w,b}(x)`: giá trị dự đoán (ví dụ: miles per gallon)

---
#### 2. Hàm mất mát - MSE (Mean Squared Error)

Với tập dữ liệu có `M` mẫu:

$$ \text{Loss}_{MSE} = \frac{1}{M} \sum_{i=1}^{M} (f_{w,b}(x^{(i)}) - y^{(i)})^2 $$

Trong đó:
- `M`: tổng số mẫu trong tập dữ liệu
- `i`: chỉ số của mẫu thứ i
- `x^{(i)}`: giá trị đặc trưng của mẫu thứ i
- `y^{(i)}`: giá trị thực tế (label) của mẫu thứ i
- `f_{w,b}(x^{(i)})`: giá trị dự đoán cho mẫu thứ i

---
#### 3. Gradient (Đạo hàm) của hàm MSE

##### 3.1. Đạo hàm theo trọng số (Weight derivative)

$$ \frac{\partial}{\partial w} \left( \frac{1}{M} \sum_{i=1}^{M} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \right) = \frac{1}{M} \sum_{i=1}^{M} (f_{w,b}(x^{(i)}) - y^{(i)}) \times 2x^{(i)} $$

**Cách tính từng bước:**
1. Với mỗi mẫu i: tính `(giá trị dự đoán - giá trị thực) * 2 * (giá trị đặc trưng x)`
2. Cộng tất cả các giá trị này lại
3. Chia tổng cho số lượng mẫu `M`

**Ký hiệu rút gọn:**

$$ \nabla w = \frac{\partial \text{MSE}}{\partial w} = \frac{2}{M} \sum_{i=1}^{M} (f_{w,b}(x^{(i)}) - y^{(i)}) \cdot x^{(i)} $$

---

##### 3.2. Đạo hàm theo độ chệch (Bias derivative)

$$ \frac{\partial}{\partial b} \left( \frac{1}{M} \sum_{i=1}^{M} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \right) = \frac{1}{M} \sum_{i=1}^{M} (f_{w,b}(x^{(i)}) - y^{(i)}) \times 2 $$

**Cách tính từng bước:**
1. Với mỗi mẫu i: tính `(giá trị dự đoán - giá trị thực) * 2`
2. Cộng tất cả các giá trị này lại
3. Chia tổng cho số lượng mẫu `M`

**Ký hiệu rút gọn:**

$$ \nabla b = \frac{\partial \text{MSE}}{\partial b} = \frac{2}{M} \sum_{i=1}^{M} (f_{w,b}(x^{(i)}) - y^{(i)}) $$

**Chú ý:** Không nhân với `x` vì đạo hàm của `b` đối với `b` là 1.

---

#### 4. Cập nhật tham số (Parameter Update)

Sau khi tính được gradient, ta cập nhật trọng số và độ chệch bằng cách **đi ngược hướng gradient**:

##### 4.1. Cập nhật trọng số

$$ w_{\text{mới}} = w_{\text{cũ}} - (\text{learning rate} \times \nabla w) $$

Hay viết đầy đủ:

$$ w_{\text{mới}} = w_{\text{cũ}} - \eta \times \frac{2}{M} \sum_{i=1}^{M} (f_{w,b}(x^{(i)}) - y^{(i)}) \cdot x^{(i)} $$

---

##### 4.2. Cập nhật độ chệch

$$ b_{\text{mới}} = b_{\text{cũ}} - (\text{learning rate} \times \nabla b) $$

Hay viết đầy đủ:

$$ b_{\text{mới}} = b_{\text{cũ}} - \eta \times \frac{2}{M} \sum_{i=1}^{M} (f_{w,b}(x^{(i)}) - y^{(i)}) $$

Trong đó:
- `η` (eta): **learning rate** (tốc độ học) - là một số dương nhỏ (ví dụ: 0.01)
- Dấu **trừ** (`-`) thể hiện việc di chuyển **ngược hướng** gradient

---

#### 5. Các khái niệm quan trọng

##### 5.1. Hội tụ (Convergence)
Mô hình **hội tụ** khi các vòng lặp tiếp theo không thể làm giảm loss đáng kể. Lúc này:
- Gradient gần bằng 0
- Loss gần như không đổi
- Đã tìm được bộ `(w, b)` tối ưu

##### 5.2. Hàm lồi (Convex function)
- **Loss của linear regression là hàm lồi** (hình chữ U hoặc bát)
- Tính chất này đảm bảo gradient descent sẽ tìm được **global minimum** (điểm thấp nhất toàn cục), không bị kẹt ở local minimum

##### 5.3. Loss curve (Đường cong loss)
- Trục x: số vòng lặp (iterations)
- Trục y: giá trị loss
- Loss giảm nhanh ở đầu, chậm dần, và phẳng khi hội tụ

---

#### 6. Tóm tắt các công thức chính

| Thành phần | Công thức |
|-----------|-----------|
| Dự đoán | $\hat{y} = w \cdot x + b$ |
| MSE Loss | $\frac{1}{M} \sum_{i=1}^{M} (\hat{y}_i - y_i)^2$ |
| Gradient theo w | $\frac{2}{M} \sum_{i=1}^{M} (\hat{y}_i - y_i) \cdot x_i$ |
| Gradient theo b | $\frac{2}{M} \sum_{i=1}^{M} (\hat{y}_i - y_i)$ |
| Cập nhật w | $w = w - \eta \cdot \nabla w$ |
| Cập nhật b | $b = b - \eta \cdot \nabla b$ |
---

**Lưu ý quan trọng:** Các công thức trên áp dụng cho **batch gradient descent** (tính gradient trên toàn bộ dữ liệu). Với SGD hoặc mini-batch, chỉ khác ở số lượng mẫu `M` dùng để tính.

### Model Convergence và Loss Curves (Đường cong hội tụ và đường cong loss)

+ Loss curve: biểu đồ thể hiện sự thay đổi của giá trị loss (trục y) theo số vòng lặp/iteration (trục x) trong quá trình huấn luyện mô hình.

+ Quá trình hội tụ (Convergence): Hội tụ là trạng thái mà mô hình đã tìm được bộ (weight, bias) tối ưu, khi đó:

  - Loss giảm rất chậm hoặc hầu như không đổi qua các vòng lặp
  - Gradient tiến dần về 0 (không còn hướng "dốc" để đi xuống)
  - Các vòng lặp thêm không cải thiện đáng kể mô hình

#### Các yếu tố ảnh hưởng đến hội tụ

| Yếu tố |	Ảnh hưởng |
| ------ | ---------- |
| Learning rate quá lớn |	Loss dao động mạnh, có thể phân kỳ (không hội tụ) |
| Learning rate quá nhỏ |	Hội tụ rất chậm, tốn nhiều vòng lặp |
| Dữ liệu nhiễu |	Loss curve có thể gai góc, khó xác định hội tụ |
| Kích thước dữ liệu lớn |	Cần nhiều vòng lặp hơn để hội tụ |
---

### Hàm lồi (Convex Functions) và "Ngọn núi một đỉnh"

Linear regression có một tính chất đặc biệt: hàm loss của nó luôn là một hàm lồi (convex function).

Hàm lồi chỉ có một điểm thấp nhất duy nhất, gọi là global minimum (điểm cực tiểu toàn cục).

---

### Hyperparameters ( Siêu tham số )

Các biến bạn điều khiển để tác động đến các khía cạnh khác nhau của quá trình huấn luyện.

Ba loại hyperparameters phổ biến:

+ Learning rate (Tốc độ học)
+ Batch size (Kích thước lô / mẻ)
+ Epochs (Số chu kỳ / số thời đại)

---

#### Learning rate (Tốc độ học)

+ Là một số thực bạn tự đặt.

+ Quyết định mức độ thay đổi của weight và bias sau mỗi bước trong gradient descent. Kích thước "bước nhảy" mỗi lần cập nhật weight/bias.

+ Công thức: `Thay đổi = learning_rate × Gradient`


##### Ảnh hưởng đến huấn luyện

| Trường hợp |	Đặc điểm |	Hậu quả |
| ---------- | -------- | -------- |
| Learning rate quá NHỎ |	Mỗi bước cải thiện rất ít |	Cần rất nhiều vòng lặp mới có thể hội tụ |
| Learning rate VỪA PHẢI | 	Giảm loss nhanh và ổn định |	Hội tụ nhanh chóng sau số vòng lặp hợp lý |
| Learning rate quá LỚN |	Loss dao động mạnh hoặc tăng vọt |	KHÔNG BAO GIỜ HỘI TỤ, thậm chí phân kỳ |

---

#### Batch size (Kích thước lô)

+ số lượng ví dụ dữ liệu (examples) được sử dụng trong một lần lặp (iteration) để tính toán gradient và cập nhật trọng số của mô hình.

##### 3 chiến lược chính về Batch Size:

- Full-batch iteration (Toàn bộ dữ liệu):

   + Sử dụng toàn bộ tập dữ liệu để tính toán gradient cho mỗi lần cập nhật
   + Gradient tính toán được là chính xác nhất cho toàn bộ tập dữ liệu
   + Nếu tập dữ liệu khổng lồ (hàng tỷ ví dụ), việc tính toán sẽ cực kỳ chậm và tốn bộ nhớ (RAM/GPU)

- Stochastic Gradient Descent (SGD):

   + Batch size = 1. Cập nhật trọng số sau mỗi một ví dụ dữ liệu duy nhất
   + Tốc độ tính toán cho mỗi bước rất nhanh
   + Gradient bị "nhiễu" (nhảy loạn xạ), đường đi đến điểm tối ưu không ổn định
 
- Mini-batch Gradient Descent (Phổ biến nhất)

   + Batch size thường nằm trong khoảng từ 10 đến 1.000 ví dụ
   + Cân bằng giữa tốc độ huấn luyện và độ ổn định
 
##### Tác động của Batch Size đến quá trình học

| Trường hợp | Ảnh hưởng |
| ---------- | ---------- |
| Batch size nhỏ | Tạo ra nhiều "nhiễu" hơn trong quá trình cập nhật. Đôi khi sự nhiễu này lại giúp mô hình thoát khỏi các "cực tiểu địa phương" (local minima) để tìm đến kết quả tốt hơn. |
| Batch size lớn | Giúp gradient ổn định hơn, nhưng nếu quá lớn, mô hình có thể bị kẹt ở những điểm không tối ưu hoặc đòi hỏi tài nguyên phần cứng vượt quá khả năng của máy tính. |

---

#### Epoch (Kỷ nguyên)

Đơn vị thời gian trong huấn luyện ML, đại diện cho một lần duy nhất mô hình đi qua toàn bộ tập dữ liệu huấn luyện (training set).

Nếu bạn thiết lập Epochs = 10, mô hình của bạn sẽ "nhìn" thấy và học từ mỗi ví dụ trong tập dữ liệu tổng cộng 10 lần.

Trong ML, mô hình không thể học hết các quy luật phức tạp chỉ sau một lần đọc dữ liệu. Việc lặp lại nhiều lần (nhiều Epochs) giúp mô hình:

 + Cập nhật trọng số (w và b) nhiều lần hơn để giảm thiểu sai số (Loss)
 + Tìm ra các mối quan hệ ẩn sâu trong dữ liệu mà lần đầu tiên có thể bỏ lỡ

---

##### Cách xác định số lượng Epochs phù hợp

Để biết khi nào nên dừng lại, các kỹ sư ML thường quan sát Loss Curve (Đồ thị mất mát):

 + Giai đoạn đầu: Đường cong Loss giảm rất nhanh (mô hình đang học tốt)
 + Giai đoạn ổn định (Convergence): Đường cong Loss bắt đầu đi ngang (mô hình đã hội tụ). Đây là lúc bạn nên dừng huấn luyện.
 + Dấu hiệu Overfitting: Nếu Loss trên tập huấn luyện tiếp tục giảm nhưng Loss trên tập kiểm tra (validation set) bắt đầu tăng lên, đó là lúc bạn đã chạy quá nhiều Epochs.

---

##### Mối quan hệ giữa Epoch, Batch Size và Iteration (Lần lặp)

+ Batch Size: Số lượng ví dụ dùng để tính toán trong 1 lần cập nhật trọng số.

+ Iteration (Lần lặp/Bước): Một lần cập nhật trọng số duy nhất.

+ Epoch: Hoàn thành việc duyệt qua toàn bộ dữ liệu.

---

###### Công thức liên hệ giữa các Siêu tham số

Công thức giúp bạn tính toán số lần cập nhật trọng số của mô hình:

$$ \text{Total Iterations} = \text{Epochs} \times \left( \frac{\text{Total Examples}}{\text{Batch Size}} \right) $$

*Trong đó:*
*   **Total Iterations**: Tổng số lần mô hình cập nhật trọng số ($w$ và $b$).
*   **Epochs**: Số lần duyệt qua toàn bộ tập dữ liệu.
*   **Total Examples**: Tổng số ví dụ trong tập dữ liệu huấn luyện.
*   **Batch Size**: Số lượng ví dụ trong một lô dữ liệu.

---

###### Quy tắc cập nhật trọng số (Gradient Descent)
Công thức để điều chỉnh các tham số sau mỗi bước lặp:

$$ w_{new} = w_{old} - \eta \cdot \nabla L $$

*Trong đó:*
*   $w$: Trọng số cần cập nhật.
*   $\eta$ (Eta): Tốc độ học (**Learning Rate**).
*   $\nabla L$: Đạo hàm của hàm mất mát (**Gradient**) theo trọng số đó.

---

#  Logistic Regression

Khác với Linear Regression (Hồi quy tuyến tính) dùng để dự đoán một giá trị số thực (như giá nhà), Logistic Regression (Hồi quy Logistic) được dùng để dự đoán xác suất của một sự kiện 

Kết quả trả về luôn nằm trong khoảng: [0,1]

## Công thức hàm Sigmoid

Phương trình dưới đây đại diện cho thành phần tuyến tính của một mô hình hồi quy logistic (logistic regression):

$$z = b + w_1x_1 + w_2x_2 + ... + w_nx_n$$

Trong đó:

*   **$z$** là đầu ra của phương trình tuyến tính, còn được gọi là **log-odds**.
*   **$b$** là độ chệch (bias).
*   Các giá trị **$w$** là các trọng số (weights) mà mô hình học được.
*   Các giá trị **$x$** là các giá trị đặc trưng (feature values) cho một ví dụ cụ thể.

Để nhận được dự đoán từ hồi quy logistic, giá trị $z$ sau đó được truyền qua hàm **sigmoid**, tạo ra một giá trị (một xác suất) nằm trong khoảng từ 0 đến 1:

$$y' = \frac{1}{1 + e^{-z}}$$

Trong đó:

*   **$y'$** là đầu ra của mô hình hồi quy logistic (xác suất dự đoán).
*   **$e$** là số Euler: một hằng số toán học $\approx 2.71828$.
*   **$z$** là đầu ra tuyến tính (đã được tính toán ở phương trình trên).

---

### Các đặc điểm quan trọng của hàm Sigmoid

Hiểu đồ thị và tính chất của hàm Sigmoid giúp bạn nắm bắt cách mô hình ra quyết định:

*   **Đầu ra bị giới hạn:** Dù $z$ có lớn hay nhỏ bao nhiêu, $y$ luôn nằm trong khoảng $(0, 1)$.
*   **Khi $z = 0$:** Giá trị $y = 0.5$. Đây là điểm trung tâm.
*   **Khi $z \to \infty$:** Giá trị $y$ tiến rất nhanh về $1$.
*   **Khi $z \to -\infty$:** Giá trị $y$ tiến rất nhanh về $0$.
*   **Hình dạng:** Đồ thị có dạng đường cong chữ **S**.

---

## Cách chuyển từ Xác suất sang Phân loại (Classification)

Sau khi có giá trị xác suất $y$ từ hàm Sigmoid, chúng ta cần một **Threshold** (Ngưỡng phân loại) để đưa ra kết luận cuối cùng (Ví dụ: 0 hoặc 1).

**Quy tắc thông thường (Ngưỡng 0.5):**
*   Nếu $y \ge 0.5 \implies$ Kết luận: **1** (Dương tính/Spam).
*   Nếu $y < 0.5 \implies$ Kết luận: **0** (Âm tính/Không phải Spam).

*Lưu ý: Ngưỡng này có thể thay đổi tùy vào mục đích bài toán (ví dụ: ngưỡng 0.7 hoặc 0.8 để khắt khe hơn).*

---

## Tại sao cần Sigmoid trong ML?

1.  **Tính xác suất:** Nó không chỉ nói "Có" hay "Không", mà nó nói cho ta biết "Khả năng cao bao nhiêu phần trăm".
2.  **Đạo hàm đẹp:** Trong toán học, hàm Sigmoid có đạo hàm rất thuận tiện cho việc tính toán **Gradient Descent** để cập nhật trọng số.
3.  **Phi tuyến tính:** Nó giúp mô hình học được các mối quan hệ không phải là đường thẳng đơn giản.

---

### Tóm tắt công thức tổng quát

Quy trình của Logistic Regression:

$$ \hat{y} = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}} $$

*   **Bước 1:** Tính giá trị tuyến tính $z = w^T x + b$.
*   **Bước 2:** Đưa $z$ qua hàm Sigmoid $\sigma(z)$ để lấy xác suất $\hat{y}$.
*   **Bước 3:** So sánh $\hat{y}$ với ngưỡng (Threshold) để phân loại.

---

## Hàm mất mát (Loss function) và Điều quy hóa (Regularization)

### Hàm mất mát: Log Loss

Trong Linear Regression, chúng ta dùng *Squared Loss* (MSE). Tuy nhiên, trong Logistic Regression, chúng ta không dùng MSE vì nó sẽ khiến việc tối ưu hóa rất khó khăn. Thay vào đó, chúng ta sử dụng **Log Loss** (còn gọi là **Binary Cross-Entropy**).

### Công thức Log Loss

$$\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(y'_i) + (1 - y_i) \log(1 - y'_i)]$$

#### Các thành phần:

*   **$N$**: Tổng số ví dụ có nhãn trong tập dữ liệu.
*   **$i$**: Chỉ số đại diện cho một ví dụ cụ thể (ví dụ: $i=1, 2, 3...$).
*   **$y_i$**: Nhãn thực tế (đáp án đúng) của ví dụ thứ $i$. Trong hồi quy logistic, giá trị này phải là **0** hoặc **1**.
*   **$y'_i$**: Giá trị dự đoán của mô hình cho ví dụ thứ $i$. Đây là một xác suất nằm trong khoảng từ **0 đến 1**, được tính toán dựa trên tập đặc trưng $x_i$.
*   **$\sum_{i=1}^{N}$**: Ký hiệu tổng, thực hiện tính toán biểu thức bên trong cho từng ví dụ rồi cộng tất cả lại.
*   **$\log$**: Hàm logarit tự nhiên.
*   **Dấu âm ($-$) ở đầu**: Được dùng để đảm bảo giá trị Loss cuối cùng là một số dương (vì log của một số < 1 luôn ra kết quả âm).

**Ý nghĩa của Log Loss:**
*   Nếu $y = 1$: Hàm mất mát chỉ còn $-\ln(y')$. Khi $y'$ tiến gần về $1$, Loss tiến về $0$. Khi $y'$ tiến gần về $0$, Loss tiến về vô cùng.
*   Nếu $y = 0$: Hàm mất mát chỉ còn $-\ln(1 - y')$. Khi $y'$ tiến gần về $0$, Loss tiến về $0$.
*   **Kết luận:** Log Loss phạt rất nặng những dự đoán "tự tin nhưng sai" (ví dụ: mô hình dự đoán xác suất là 0.99 nhưng nhãn thực tế là 0).

---

### Sự phức tạp của mô hình và Overfitting

Trong Logistic Regression, nếu chúng ta không kiểm soát, các trọng số ($w$) có thể tiến tới vô cùng khi mô hình cố gắng khớp hoàn toàn (perfect fit) với mọi điểm dữ liệu trong tập huấn luyện. Điều này dẫn đến **Overfitting** (quá khớp).

Để ngăn chặn điều này, chúng ta cần giảm độ phức tạp của mô hình thông qua **Regularization (Điều quy hóa)**.

---

### Điều quy hóa L2 (L2 Regularization)

Đây là kỹ thuật phổ biến nhất để phạt các trọng số quá lớn. Thay vì chỉ tối ưu hóa để giảm thiểu Loss, chúng ta tối ưu hóa một hàm mới:

$$\text{Minimize: } \text{Log Loss}(Data, Model) + \text{Complexity}(Model)$$

Trong **L2 Regularization**, độ phức tạp được tính bằng tổng bình phương của các trọng số:
$$\text{Complexity} = \lambda \sum w^2$$

*   **$\lambda$ (Lambda):** Là một siêu tham số (hyperparameter) điều khiển mức độ phạt.
*   **Mục tiêu:** Giữ cho các trọng số nhỏ lại, giúp mô hình ổn định hơn và khái quát hóa tốt hơn với dữ liệu mới.

---

### Tác động của siêu tham số Lambda ($\lambda$)
Việc chọn $\lambda$ cực kỳ quan trọng để cân bằng giữa việc học dữ liệu và giữ mô hình đơn giản:

| Giá trị $\lambda$ | Tác động | Kết quả |
| :--- | :--- | :--- |
| **Quá thấp ($\approx 0$)** | Ít hoặc không phạt trọng số. | Dễ bị **Overfitting**. |
| **Vừa đủ** | Cân bằng giữa Loss và độ đơn giản. | Mô hình tốt nhất (Generalization). |
| **Quá cao** | Trọng số bị ép về gần bằng 0. | Dễ bị **Underfitting** (Mô hình quá đơn giản). |

---

### Tổng hợp:
1.  **Log Loss** là thước đo chuẩn xác cho các bài toán phân loại xác suất.
2.  **Regularization** không phải là tùy chọn, nó là **bắt buộc** để mô hình không bị "học vẹt" dữ liệu.
3.  Khi huấn luyện mô hình, nếu thấy trọng số ($w$) tăng quá lớn, hãy tăng giá trị **Lambda**.
4.  Luôn theo dõi Loss trên cả tập Training và Validation để tìm ra điểm $\lambda$ tối ưu.

---

# Classification

Nhiệm vụ dự đoán một ví dụ dữ liệu thuộc về lớp (class) hoặc danh mục nào trong một tập hợp cho trước.

## Threshold ( Ngưỡng phân loại )

Hồi quy Logistic trả về một xác suất (ví dụ: 0.7 hoặc 0.2). Tuy nhiên, để đưa ra quyết định cuối cùng (ví dụ: Email này là Spam hay Không), chúng ta cần một giá trị "điểm cắt", gọi là Ngưỡng phân loại (Classification Threshold).

---

### Quy tắc ánh xạ
Giả sử chúng ta chọn một ngưỡng là $T$:
*   Nếu Xác suất $y' \ge T \implies$ Phân loại là **Dương tính (Positive)**.
*   Nếu Xác suất $y' < T \implies$ Phân loại là **Âm tính (Negative)**.

*Ví dụ: Với ngưỡng $T = 0.5$:*
*   $y' = 0.6 \implies$ Spam.
*   $y' = 0.4 \implies$ Không phải Spam.

---

### Ngưỡng 0.5 có phải luôn là tốt nhất?
**Câu trả lời là KHÔNG.** Mặc dù 0.5 là ngưỡng mặc định phổ biến, nhưng việc chọn ngưỡng phụ thuộc hoàn toàn vào **tác động thực tế** của các sai lầm trong bài toán của bạn.

#### Trường hợp 1: Cần ngưỡng THẤP (ví dụ: 0.1)
Khi việc **bỏ sót** một ca dương tính là cực kỳ nguy hiểm.
*   *Ví dụ:* Chẩn đoán ung thư. Bạn thà dự đoán nhầm một người khỏe mạnh là có bệnh (để kiểm tra kỹ hơn) còn hơn là bỏ sót một người thực sự có bệnh.

#### Trường hợp 2: Cần ngưỡng CAO (ví dụ: 0.9)
Khi việc **kết luận sai** một ca dương tính gây ra hậu quả phiền toái lớn.
*   *Ví dụ:* Chặn email spam. Người dùng sẽ rất bực mình nếu một email quan trọng của sếp bị tống vào hòm thư rác. Bạn chỉ nên đánh dấu là Spam khi mô hình cực kỳ chắc chắn (xác suất > 0.9).

---

### Tác động của việc thay đổi ngưỡng

Việc điều chỉnh ngưỡng sẽ làm thay đổi các chỉ số đánh giá mô hình (Metrics):
*   **Hạ thấp ngưỡng:** Bạn sẽ thu được nhiều kết quả Dương tính hơn $\implies$ Tăng khả năng bắt được mọi trường hợp đúng (**Recall** tăng), nhưng dễ bị nhầm (**Precision** giảm).
*   **Nâng cao ngưỡng:** Bạn sẽ khắt khe hơn $\implies$ Những gì bạn kết luận là Dương tính sẽ có độ tin cậy cao hơn (**Precision** tăng), nhưng bạn sẽ bỏ lỡ nhiều trường hợp thực tế (**Recall** giảm).

---

### Tóm tắt:
1.  **Thresholding** là bước chuyển đổi từ xác suất sang nhãn loại.
2.  **Ngưỡng là một siêu tham số (Hyperparameter):** Bạn không học nó bằng Gradient Descent, mà bạn phải tự chọn dựa trên mục đích bài toán.
3.  **Không bao giờ đánh giá mô hình chỉ bằng 1 ngưỡng duy nhất:** Bạn cần nhìn vào bức tranh toàn cảnh (như biểu đồ ROC hay Precision-Recall) để hiểu hiệu suất của mô hình ở các ngưỡng khác nhau.

---

## Confusion Matrix (Ma trận nhầm lẫn)

**Confusion Matrix (Ma trận nhầm lẫn)** là một công cụ cực kỳ quan trọng để đánh giá hiệu suất của một mô hình phân loại. Nó cho bạn thấy không chỉ mô hình sai bao nhiêu, mà còn là **sai như thế nào**.

Confusion Matrix là một bảng $2 \times 2$ (đối với phân loại nhị phân) so sánh giữa **Giá trị thực tế** và **Giá trị dự đoán**.

### Cấu trúc ma trận

| | Thực tế là Dương tính (Positive) | Thực tế là Âm tính (Negative) |
| :--- | :---: | :---: |
| **Dự đoán là Dương tính** | **TP** (True Positive) | **FP** (False Positive) |
| **Dự đoán là Âm tính** | **FN** (False Negative) | **TN** (True Negative) |

---

### Giải thích 4 thành phần chính (Rất quan trọng)

*   **TP (True Positive - Dương tính thật):** Mô hình dự đoán là "Có" và thực tế là "Có". (Ví dụ: Dự đoán có bệnh, thực tế có bệnh - **Đúng**)
*   **TN (True Negative - Âm tính thật):** Mô hình dự đoán là "Không" và thực tế là "Không". (Ví dụ: Dự đoán không bệnh, thực tế khỏe mạnh - **Đúng**)
*   **FP (False Positive - Dương tính giả):** Mô hình dự đoán là "Có" nhưng thực tế là "Không".
    *   *Còn gọi là: Sai lầm loại I (Type I Error).*
    *   *Hệ quả:* Gây ra báo động giả (Ví dụ: Người khỏe mạnh bị kết luận có bệnh - **"Oan sai"**).
*   **FN (False Negative - Âm tính giả):** Mô hình dự đoán là "Không" nhưng thực tế là "Có".
    *   *Còn gọi là: Sai lầm loại II (Type II Error).*
    *   *Hệ quả:* Bỏ sót trường hợp quan trọng (Ví dụ: Người có bệnh bị kết luận là khỏe mạnh - **"Lọt lưới"**).

---

### Tại sao cần Confusion Matrix?
Tại sao không dùng mỗi **Accuracy (Độ chính xác tổng thể)**? 

**Ví dụ:** Bạn có 100 người, trong đó 99 người khỏe và 1 người bị ung thư.
*   Nếu mô hình của bạn luôn dự đoán "Khỏe mạnh" cho tất cả mọi người, Accuracy sẽ là **99%**.
*   Nhìn qua thì 99% là rất cao, nhưng thực tế mô hình này **vô dụng** vì nó bỏ sót ca ung thư duy nhất (FN = 1).
*   **Confusion Matrix** sẽ vạch trần điều này bằng cách cho bạn thấy FN = 1 và TP = 0.

---

### Các chỉ số rút ra từ Ma trận nhầm lẫn
Từ 4 giá trị TP, TN, FP, FN, ta tính được các chỉ số quan trọng khác:

1.  **Accuracy (Độ chính xác):** Tỷ lệ dự đoán đúng trên tổng số ca.

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

2. **False positive rate (Tỷ lệ dương tính giả):** Tỷ lệ giữa tất cả các trường hợp thực tế là âm tính nhưng bị phân loại sai thành dương tính. Chỉ số này còn được gọi là xác suất báo động giả (probability of false alarm)

$$\text{FPR} = \frac{FP}{FP + TN}$$

3.  **Precision (Độ chính xác trên tập dự đoán dương):** Trong những ca mô hình bảo là "Có", có bao nhiêu ca thực sự là "Có"? (Tránh "Oan sai").

$$\text{Precision} = \frac{TP}{TP + FP}$$

4.  **Recall (Độ bao phủ/Độ nhạy):** Trong số những ca thực tế là "Có", mô hình bắt được bao nhiêu ca? (Tránh "Lọt lưới").

$$\text{Recall} = \frac{TP}{TP + FN}$$

---

### Mẹo ghi nhớ:
*   **True/False:** Nói về việc dự đoán đó **Đúng** hay **Sai**.
*   **Positive/Negative:** Nói về **tên nhãn** mà mô hình đã chọn.
*   Cứ thấy **True** là bạn đã làm đúng, thấy **False** là bạn đã làm sai.

---

### Lựa chọn chỉ số và Sự đánh đổi

Chỉ số (metric) mà bạn chọn ưu tiên khi đánh giá mô hình và chọn ngưỡng (threshold) phụ thuộc vào chi phí, lợi ích và rủi ro của từng bài toán cụ thể. 

| Chỉ số | Hướng dẫn sử dụng |
| :--- | :--- |
| **Accuracy** (Độ chính xác) | - Sử dụng như một chỉ số sơ bộ để theo dõi tiến trình huấn luyện hoặc sự hội tụ của mô hình đối với các **tập dữ liệu cân bằng**.<br>- Đối với hiệu suất thực tế của mô hình, chỉ nên dùng khi kết hợp với các chỉ số khác.<br>- **Tránh sử dụng** cho các tập dữ liệu không cân bằng (imbalanced datasets). Hãy cân nhắc sử dụng chỉ số khác. |
| **Recall** (Tỷ lệ dương tính thật) | - Sử dụng khi cái giá phải trả cho **Âm tính giả** (False Negatives - lọt lưới) lớn hơn so với Dương tính giả. |
| **False positive rate** (Tỷ lệ dương tính giả) | - Sử dụng khi cái giá phải trả cho **Dương tính giả** (False Positives - báo động nhầm) lớn hơn so với Âm tính giả. |
| **Precision** (Độ xác thực) | - Sử dụng khi việc các dự đoán Dương tính phải đảm bảo độ chính xác là **cực kỳ quan trọng**. |

---

## Đường cong ROC và Chỉ số AUC

### Đường cong ROC (Receiver Operating Characteristic)

Đường cong ROC là một đồ thị thể hiện hiệu suất của mô hình phân loại ở **tất cả các ngưỡng phân loại khác nhau**.

*   **Trục tung (Y):** **TPR** (True Positive Rate) — chính là **Recall**.
    $$TPR = \frac{TP}{TP + FN}$$
*   **Trục hoành (X):** **FPR** (False Positive Rate).
    $$FPR = \frac{FP}{FP + TN}$$

**Cách hoạt động:** Khi bạn hạ thấp ngưỡng phân loại, mô hình sẽ dự đoán nhiều mục là Dương tính hơn $\implies$ cả TP và FP đều tăng $\implies$ đường cong sẽ chạy từ điểm $(0,0)$ dần lên $(1,1)$.

---

### AUC (Area Under the ROC Curve)

**AUC** là diện tích nằm dưới đường cong ROC. Chỉ số này cung cấp một giá trị duy nhất (từ 0 đến 1) để tóm tắt toàn bộ hiệu suất của mô hình.

*   **Ý nghĩa xác suất:** AUC đại diện cho xác suất mà mô hình sẽ xếp hạng một ví dụ Dương tính ngẫu nhiên **cao hơn** một ví dụ Âm tính ngẫu nhiên.
*   **Các giá trị AUC phổ biến:**
    *   **AUC = 1.0:** Mô hình hoàn hảo (phân loại đúng 100%).
    *   **AUC = 0.5:** Mô hình dự đoán ngẫu nhiên (giống như tung đồng xu, không có giá trị phân loại).
    *   **AUC = 0.0:** Mô hình sai hoàn toàn (luôn dự đoán ngược lại).

---

### Tại sao AUC lại quan trọng?

AUC có hai ưu điểm lớn khiến nó rất được ưa chuộng:

1.  **Bất biến với ngưỡng (Threshold-invariant):** AUC đo lường chất lượng dự đoán của mô hình mà không cần quan tâm bạn chọn ngưỡng nào (0.5, 0.3 hay 0.9). Nó đánh giá khả năng **phân tách** các lớp của mô hình.
2.  **Bất biến với quy mô (Scale-invariant):** AUC đo lường mức độ **xếp hạng** của các dự đoán thay vì giá trị tuyệt đối. (Ví dụ: Nếu mô hình dự đoán xác suất là 0.2 và 0.4, AUC vẫn cho kết quả tương tự như khi mô hình dự đoán 0.8 và 0.9, miễn là thứ tự xếp hạng không đổi).

---

### Những lưu ý (Caveats) khi dùng AUC

Mặc dù rất mạnh mẽ, AUC vẫn có những hạn chế:

*   **Không phản ánh chi phí sai lầm:** AUC coi trọng mọi lỗi sai như nhau. Trong thực tế, nếu cái giá của Dương tính giả (FP) quá đắt so với Âm tính giả (FN), AUC có thể không phản ánh đúng thực tế bạn cần.
*   **Vấn đề về dữ liệu mất cân bằng:** Trong các bài toán có dữ liệu cực kỳ lệch (ví dụ: tỷ lệ bệnh hiếm 1/10.000), AUC đôi khi vẫn cho con số rất cao dù mô hình thực tế không hoạt động tốt trên lớp thiểu số.

---

## Độ lệch dự đoán (Prediction Bias)

### Định nghĩa

**Prediction Bias** là một đại lượng đo lường khoảng cách giữa giá trị trung bình của các dự đoán từ mô hình so với giá trị trung bình thực tế của các nhãn trong tập dữ liệu.

**Công thức:**
$$\text{Prediction Bias} = \text{Average of Predictions} - \text{Average of Labels}$$

*Lưu ý: Độ lệch này khác với khái niệm "Bias" ($b$) trong phương trình tuyến tính $y = wx + b$.*

---

### Ý nghĩa của các giá trị Bias
*   **Bias = 0:** Một mô hình lý tưởng nên có độ lệch dự đoán bằng 0. Điều này có nghĩa là trung bình các dự đoán khớp hoàn toàn với trung bình các giá trị thực tế.
*   **Bias dương (> 0):** Mô hình đang **dự đoán quá cao** (overestimating). Ví dụ: trung bình dự đoán là 0.7 nhưng thực tế chỉ là 0.5.
*   **Bias âm (< 0):** Mô hình đang **dự đoán quá thấp** (underestimating).

---

### Cách chẩn đoán: Chia nhóm (Bucketing)
Đừng chỉ nhìn vào độ lệch tổng thể của toàn bộ tập dữ liệu, vì các sai số có thể triệt tiêu lẫn nhau làm bạn lầm tưởng mô hình đã tốt. 

**Kỹ thuật:** Chia dữ liệu thành các "nhóm" (buckets). Ví dụ:
*   Chia theo các khoảng giá trị dự đoán (từ 0 đến 0.1, 0.1 đến 0.2...).
*   Trong mỗi nhóm, tính toán Prediction Bias.
*   **Biểu đồ hiệu chuẩn (Calibration Plot):** Nếu đường dự đoán khớp với đường 45 độ của các nhãn thực tế, mô hình đã được hiệu chuẩn tốt.

---

### Nguyên nhân gây ra Prediction Bias
Nếu mô hình có độ lệch lớn, nguyên nhân thường do:
*   **Tập đặc trưng (Features) chưa đầy đủ:** Có các yếu tố quan trọng ảnh hưởng đến kết quả nhưng chưa được đưa vào mô hình.
*   **Dữ liệu nhiễu (Noisy data):** Tập huấn luyện chứa các ví dụ bị sai nhãn hoặc không đại diện cho thực tế.
*   **Quy trình huấn luyện bị lỗi (Biased pipeline):** Cách lấy mẫu dữ liệu không công bằng.
*   **Điều quy hóa (L2 Regularization) quá mạnh:** Khi bạn phạt các trọng số quá nặng, mô hình có xu hướng kéo các dự đoán về gần mức trung bình chung, dẫn đến phát sinh độ lệch.

---

### Cảnh báo quan trọng: Đừng "sửa ngọn"
Khi phát hiện có Prediction Bias, nhiều người mắc sai lầm là thêm một **"lớp hiệu chuẩn" (calibration layer)** ở đầu ra để điều chỉnh kết quả dự đoán (ví dụ: cộng thêm một hằng số vào kết quả cuối).

**Tại sao không nên làm vậy?**
1.  Bạn đang sửa triệu chứng chứ không phải sửa nguyên nhân gốc rễ.
2.  Mô hình sẽ trở nên lỗi thời nhanh chóng nếu dữ liệu đầu vào thay đổi.
3.  Thay vào đó, hãy tập trung vào việc **cải thiện đặc trưng** hoặc **kiểm tra lại dữ liệu huấn luyện**.

---

Dưới đây là tóm tắt nội dung cốt lõi về **Phân loại đa lớp (Multi-class Classification)** từ Google Machine Learning Crash Course:

---

## Phân loại đa lớp (Multi-class Classification)

Phân loại đa lớp là bài toán dự đoán một ví dụ thuộc về **một trong số nhiều lớp** (nhiều hơn 2 lớp).
*   *Ví dụ:* Nhận diện chữ số viết tay (từ 0 đến 9), phân loại các loài động vật (chó, mèo, chim, cá).

---

### Hai phương pháp tiếp cận chính

#### a. Một-đối-tất cả (One-vs.-All)
Phương pháp này chia bài toán đa lớp thành nhiều bài toán phân loại nhị phân.
*   **Cách làm:** Với $N$ lớp, bạn huấn luyện $N$ bộ phân loại nhị phân riêng biệt. 
*   *Ví dụ:* Để phân loại Chó, Mèo, Chim:
    1. Bộ phân loại 1: Chó hay không phải Chó?
    2. Bộ phân loại 2: Mèo hay không phải Mèo?
    3. Bộ phân loại 3: Chim hay không phải Chim?
*   **Kết quả:** Khi dự đoán, ví dụ đó được đưa qua cả 3 bộ lọc, bộ nào có xác suất cao nhất sẽ được chọn.

#### b. Softmax (Phổ biến nhất)
Thay vì dùng nhiều mô hình nhị phân, chúng ta dùng một mô hình duy nhất có lớp đầu ra là **Softmax**. Softmax mở rộng ý tưởng của hàm Sigmoid từ nhị phân sang đa lớp.

*   **Đặc điểm của Softmax:**
    1. Trả về một **phân phối xác suất** cho tất cả các lớp.
    2. Tổng xác suất của tất cả các lớp luôn **bằng 1.0**.
    3. Công thức: $\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$

---

### Phân biệt các loại Softmax
*   **Full Softmax:** Tính toán xác suất cho tất cả các lớp có thể. Hiệu quả khi số lượng lớp ít (ví dụ: < 1000 lớp).
*   **Candidate Sampling (Lấy mẫu ứng viên):** Chỉ tính toán xác suất cho nhãn đúng và một mẫu ngẫu nhiên các nhãn sai. Thường dùng khi số lượng lớp cực lớn (ví dụ: hàng triệu từ trong mô hình ngôn ngữ) để tiết kiệm tài nguyên.

---

### Single-label vs. Multi-label (Quan trọng)
Bạn cần xác định rõ mục tiêu bài toán để chọn hàm kích hoạt (activation function) phù hợp:

*   **Phân loại đơn nhãn (Single-label):** Một ví dụ chỉ thuộc về **duy nhất một lớp**.
    *   *Ví dụ:* Một bức ảnh chỉ có thể là Chó **hoặc** Mèo.
    *   **Giải pháp:** Dùng **Softmax**.
*   **Phân loại đa nhãn (Multi-label):** Một ví dụ có thể thuộc về **nhiều lớp cùng lúc**.
    *   *Ví dụ:* Một bức ảnh vừa có Chó, vừa có Mèo.
    *   **Giải pháp:** KHÔNG dùng Softmax. Thay vào đó, dùng nhiều hàm **Sigmoid** riêng biệt cho mỗi lớp.

---

### Ghi nhớ cho ML:
1.  Nếu tổng xác suất của các lựa chọn phải bằng 100% $\rightarrow$ Dùng **Softmax**.
2.  Nếu các lựa chọn độc lập với nhau (một thứ có thể là nhiều thứ) $\rightarrow$ Dùng nhiều **Sigmoid**.
3.  **One-vs.-All** tốn tài nguyên hơn Softmax vì phải chạy nhiều mô hình độc lập, nhưng dễ giải thích hơn trong một số trường hợp.
4.  Khi huấn luyện Softmax, hàm mất mát thường được dùng là **Multi-class Cross Entropy**.

---

# Vector đặc trưng (Feature Vectors)

### Định nghĩa

**Vector đặc trưng** là một danh sách các con số đại diện cho các đặc trưng (features) của một ví dụ (example) duy nhất. Đây là dữ liệu đầu vào thực tế mà mô hình Machine Learning sẽ dùng để tính toán.

*   *Ví dụ:* Để dự đoán giá nhà, vector đặc trưng có thể là: $[2000, 3, 2]$ (tương ứng với diện tích, số phòng ngủ, số phòng tắm).

### Tại sao cần Vector đặc trưng?
Mô hình ML thực chất là các phép toán (nhân ma trận, cộng số). Do đó, mọi dữ liệu dù là hình ảnh, văn bản hay âm thanh đều phải được chuyển đổi thành các con số trong một vector trước khi đưa vào huấn luyện.

---

### Các quy tắc thiết kế Vector đặc trưng tốt

Để mô hình học hiệu quả, dữ liệu số trong vector cần tuân thủ các nguyên tắc sau:

#### a. Tránh các "giá trị ma thuật" (Magic Values)
*   **Sai:** Dùng các con số đặc biệt để đánh dấu dữ liệu thiếu. Ví dụ: đặt `score = -1` nếu học sinh bỏ thi. Mô hình sẽ hiểu lầm rằng -1 là một giá trị toán học thực tế và dùng nó để nhân/chia.
*   **Đúng:** Sử dụng một đặc trưng phụ dạng Boolean (ví dụ: `is_missing_score = True`) hoặc dùng các kỹ thuật xử lý dữ liệu thiếu (như điền giá trị trung bình).

#### b. Đơn vị phải rõ ràng và thống nhất
*   Mô hình không biết đơn vị của con số là gì (mét, dặm hay giây). 
*   Hãy đảm bảo mọi giá trị trong cùng một đặc trưng đều dùng chung một đơn vị đo lường.

#### c. Làm sạch dữ liệu (Data Scrubbing)
Trước khi đưa vào vector, cần xử lý các lỗi dữ liệu phổ biến:
*   **Outliers (Giá trị ngoại lai):** Những con số cực lớn hoặc cực nhỏ bất thường có thể làm chệch hướng huấn luyện.
*   **Dữ liệu rác:** Loại bỏ các ví dụ bị sai nhãn hoặc thiếu quá nhiều thông tin.

---

### Các bước biến đổi dữ liệu (Feature Transformation)
Để các con số trong vector "thân thiện" hơn với mô hình, chúng ta thường áp dụng:

1.  **Scaling (Thay đổi quy mô):** Đưa các đặc trưng về cùng một khoảng giá trị (ví dụ từ 0 đến 1 hoặc từ -1 đến 1). Điều này giúp Gradient Descent hội tụ nhanh hơn.
2.  **Handling Outliers:** Sử dụng kỹ thuật như **Clipping** (giới hạn giá trị tối đa/tối thiểu) để giảm ảnh hưởng của các điểm dữ liệu cực đoan.
3.  **Binning (Chia thùng):** Biến một giá trị số liên tục thành các khoảng (ví dụ: tuổi từ 10-20, 20-30) nếu mối quan hệ giữa đặc trưng và nhãn không phải là tuyến tính đơn giản.

---

Dưới đây là tóm tắt nội dung cốt lõi về **Chuẩn hóa dữ liệu (Normalization)** từ Google Machine Learning Crash Course. Đây là một trong những bước tiền xử lý quan trọng nhất đối với dữ liệu số.

---

## Chuẩn hóa dữ liệu (Normalization)

### Tại sao cần chuẩn hóa?

Khi các đặc trưng (features) có các phạm vi giá trị khác nhau (ví dụ: tuổi từ 0-100, nhưng thu nhập từ 0-1.000.000.000), mô hình sẽ gặp khó khăn:
*   **Gradient Descent hội tụ chậm:** Thuật toán sẽ mất rất nhiều thời gian để "nhảy" về điểm tối ưu vì các trọng số có quy mô quá lệch nhau.
*   **Lỗi mất mát (Loss) bị chi phối:** Các đặc trưng có giá trị lớn sẽ ảnh hưởng đến hàm mất mát nhiều hơn so với các đặc trưng có giá trị nhỏ, dù chúng có thể quan trọng như nhau.
*   **Lỗi NaN:** Giá trị quá lớn có thể gây tràn bộ nhớ khi tính toán (exploding gradients).

---

### Các kỹ thuật chuẩn hóa phổ biến

#### a. Linear Scaling (Thang đo tuyến tính)
Đưa dữ liệu về một khoảng cố định, thường là **[0, 1]** hoặc **[-1, 1]**.
*   **Công thức:** $x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$
*   **Khi nào dùng:** Khi dữ liệu phân phối tương đối đều trong một khoảng và bạn biết rõ giá trị Min/Max.

#### b. Z-score (Standardization - Chuẩn hóa điểm Z)
Biến đổi dữ liệu sao cho giá trị trung bình (mean) bằng **0** và độ lệch chuẩn (std) bằng **1**.
*   **Công thức:** $x' = \frac{x - \mu}{\sigma}$
*   **Ưu điểm:** Giúp xử lý các giá trị ngoại lai (outliers) tốt hơn Linear Scaling vì nó không ép toàn bộ dữ liệu vào một khoảng hẹp nếu có một vài điểm quá xa.

#### c. Log Scaling (Thang đo Logarit)
Dùng hàm Log để "nén" các dữ liệu có phạm vi cực rộng hoặc bị lệch nặng (Power Law distribution).
*   **Công thức:** $$x' = \log(x + 1)$$
*   **Ví dụ:** Số lượng view của video (có cái 10 view, có cái 1 tỷ view).
*   **Tác dụng:** Biến các giá trị cách biệt khổng lồ thành các khoảng cách nhỏ hơn, giúp mô hình dễ học hơn.

#### d. Clipping (Cắt cụt)
Giới hạn các giá trị ngoại lai (outliers) tại một ngưỡng cố định.
*   **Cách làm:** Nếu giá trị > ngưỡng Max, đặt nó bằng Max. Nếu < ngưỡng Min, đặt bằng Min.
*   **Tác dụng:** Loại bỏ ảnh hưởng tiêu cực của các giá trị cực đoan mà không cần xóa bỏ cả dòng dữ liệu đó.
*   **Công thức:** $$x' = \max(V_{min}, \min(x, V_{max}))$$
---

### Khi nào thì dùng kỹ thuật nào?

| Đặc điểm dữ liệu | Kỹ thuật khuyên dùng |
| :--- | :--- |
| Dữ liệu nằm trong khoảng hẹp, biết rõ Min/Max | **Linear Scaling** |
| Dữ liệu có nhiều giá trị ngoại lai (outliers) | **Z-score** hoặc **Clipping** |
| Dữ liệu bị lệch (ví dụ: thu nhập, dân số) | **Log Scaling** |
| Dữ liệu có phân phối hình chuông (Normal distribution) | **Z-score** |

---

### Ghi nhớ cho ML:
1.  **Mục tiêu cuối cùng:** Đưa mọi đặc trưng về cùng một "quy mô" (thường quanh khoảng -1 đến 1 hoặc 0 đến 1).
2.  **Không có công thức vạn năng:** Bạn nên thử nghiệm nhiều cách chuẩn hóa khác nhau và quan sát biểu đồ Loss để chọn cách tốt nhất.
3.  **Thống nhất:** Bạn phải dùng cùng một thông số chuẩn hóa (như Mean hay Std của tập Train) để áp dụng cho tập Test và dữ liệu thực tế sau này.

Việc chuẩn hóa tốt có thể giúp mô hình của bạn chạy nhanh hơn gấp nhiều lần và đạt độ chính xác cao hơn!
 
---

## Binning (Bucketing) - Chia nhóm dữ liệu

### Định nghĩa
**Binning** là kỹ thuật biến đổi một đặc trưng số liên tục thành các đặc trưng phân loại (categorical) bằng cách chia dải giá trị của nó thành các khoảng rời rạc (gọi là các "thùng" hoặc "nhóm").

*   *Ví dụ:* Thay vì để đặc trưng "Tuổi" chạy từ 0 đến 100, ta chia thành các nhóm: [0-10], [11-20], [21-30]...

---

### Tại sao cần Binning?
Mô hình tuyến tính (Linear models) học theo quy tắc: nếu giá trị đặc trưng tăng, kết quả dự đoán sẽ tăng hoặc giảm tỉ lệ thuận ($y = wx$). Tuy nhiên, thực tế thường phức tạp hơn:

*   **Xử lý quan hệ phi tuyến:** Có những đặc trưng mà mối quan hệ của nó với nhãn không phải là đường thẳng. 
    *   *Ví dụ:* Giá nhà không giảm đều theo vĩ độ, mà có thể rất đắt ở một vài vĩ độ nhất định (trung tâm thành phố) và rẻ ở những vĩ độ khác.
*   **Tăng tính linh hoạt:** Binning cho phép mô hình học một trọng số ($w$) riêng biệt cho mỗi khoảng giá trị, thay vì dùng chung một trọng số cho toàn bộ dải số.

---

### Cách thực hiện
Sau khi chia thùng, mỗi khoảng giá trị sẽ được biến đổi bằng **One-hot encoding**. 

*Ví dụ đặc trưng Vĩ độ:*
*   Nếu vĩ độ nằm trong khoảng 34-35: Vector là `[1, 0, 0]`
*   Nếu vĩ độ nằm trong khoảng 35-36: Vector là `[0, 1, 0]`
*   Nếu vĩ độ nằm trong khoảng 36-37: Vector là `[0, 0, 1]`

---

### Các chiến lược chia thùng phổ biến
1.  **Fixed Boundaries (Ranh giới cố định):** Bạn tự chọn các điểm cắt dựa trên kiến thức thực tế (ví dụ: chia độ tuổi theo các cột mốc pháp lý).
2.  **Quantile Binning (Chia theo phân vị):** Chia sao cho số lượng ví dụ trong mỗi thùng là xấp xỉ bằng nhau. Điều này giúp tránh tình trạng có thùng quá nhiều dữ liệu, có thùng lại trống rỗng.

---

### Ghi nhớ cho ML:
*   Binning biến một cột đặc trưng thành **nhiều cột** mới (tương ứng với số thùng).
*   Đừng chia quá nhiều thùng: Nếu số lượng thùng quá lớn, mô hình dễ bị **Overfitting** vì mỗi thùng có quá ít dữ liệu để học.
*   Binning là giải pháp mạnh mẽ để giúp các mô hình tuyến tính đơn giản học được các quy luật phức tạp.

---

## Data Scrubbing (Làm sạch dữ liệu)

### Tầm quan trọng
Dữ liệu trong thế giới thực thường bị lỗi, thiếu hoặc nhiễu. Scrubbing là quá trình lọc bỏ và sửa chữa các lỗi này để đảm bảo mô hình không học những quy luật sai trái.

### Xử lý dữ liệu thiếu (Missing Values)
Khi một đặc trưng bị trống, bạn có các lựa chọn sau:
*   **Xóa bỏ (Omission):** Xóa toàn bộ dòng dữ liệu đó (chỉ nên làm khi số lượng dữ liệu thiếu rất ít).
*   **Điền giá trị (Imputation):** 
    *   Điền bằng giá trị trung bình (mean) hoặc trung vị (median) của cột đó.
    *   Điền bằng một giá trị hằng số (ví dụ: 0 hoặc -1) và tạo thêm một cột phụ "is_missing" (Boolean) để báo cho mô hình biết giá trị này vốn dĩ bị thiếu.
*   **Bỏ đặc trưng:** Nếu một cột bị thiếu quá nhiều (ví dụ > 50%), hãy cân nhắc bỏ luôn cột đó.

### Xử lý giá trị ngoại lai (Outliers)
Những giá trị quá lớn hoặc quá nhỏ bất thường có thể làm chệch hướng mô hình.
*   **Nhận diện:** Sử dụng biểu đồ Histogram hoặc Scatter plot để thấy các điểm dữ liệu "đi lạc".
*   **Xử lý:** 
    *   Sử dụng **Clipping** để ép các giá trị này về một ngưỡng hợp lý.
    *   Nếu đó là dữ liệu sai do lỗi nhập liệu (ví dụ: tuổi = 200), hãy xóa bỏ dòng đó.

### Kiểm tra tính nhất quán (Consistency)
Cần rà soát các vấn đề về logic trong dữ liệu:
*   **Đơn vị:** Đảm bảo tất cả dữ liệu trong một cột dùng chung một đơn vị (ví dụ: tất cả là mét, không lẫn lộn với dặm).
*   **Nhãn sai (Bad labels):** Kiểm tra xem có trường hợp nào cùng một loại dữ liệu nhưng bị gắn nhãn khác nhau không (ví dụ: "Spam" và "S.pam").
*   **Dữ liệu trùng lặp (Duplicates):** Xóa các dòng dữ liệu giống hệt nhau vì chúng có thể làm mô hình quá coi trọng các ví dụ đó (gây Overfitting).

### Dữ liệu không đáng tin cậy
Hãy đặt câu hỏi cho các đặc trưng:
*   Đặc trưng này có tỷ lệ giá trị 0 quá cao không? (Ví dụ: cột "Số lần mua hàng" mà 99% là 0 thì mô hình rất khó học).
*   Đặc trưng này có thay đổi theo thời gian không? (Ví dụ: dữ liệu thu thập từ 10 năm trước có thể không còn đúng ở hiện tại).

---

### Ghi nhớ cho ML:
*   Luôn dành thời gian **vẽ biểu đồ** để quan sát dữ liệu trước khi làm sạch.
*   **Scrubbing không phải là làm một lần:** Bạn thường phải lặp đi lặp lại quá trình này khi phát hiện mô hình dự đoán sai ở những nhóm dữ liệu nhất định.
*   Dữ liệu sạch quan trọng hơn thuật toán phức tạp. Một mô hình đơn giản chạy trên dữ liệu sạch luôn tốt hơn mô hình phức tạp chạy trên dữ liệu bẩn.

---

## Biến đổi đa thức (Polynomial Transforms)

### Định nghĩa
**Biến đổi đa thức** là kỹ thuật tạo ra các đặc trưng mới bằng cách nâng lũy thừa các đặc trưng hiện có (ví dụ: $x^2, x^3$) hoặc nhân các đặc trưng với nhau.

### Tại sao cần Biến đổi đa thức?
Các mô hình tuyến tính (Linear Regression) mặc định chỉ có thể học các đường thẳng. Tuy nhiên, dữ liệu thực tế thường có mối quan hệ **phi tuyến** (đường cong).
*   **Vấn đề:** Một đường thẳng không thể khớp với các điểm dữ liệu phân bố theo hình parabol hoặc hình sóng.
*   **Giải pháp:** Bằng cách thêm các đặc trưng đa thức, chúng ta cho phép mô hình tuyến tính "uốn cong" đường dự đoán để khớp với dữ liệu hơn.

### Ví dụ về công thức
Giả sử bạn có một đặc trưng gốc là $x$ (diện tích nhà):

*   **Mô hình tuyến tính bậc 1:**
    $$y = w_1x + b$$
    *(Kết quả là một đường thẳng)*

*   **Mô hình đa thức bậc 2:**
    $$y = w_1x + w_2x^2 + b$$
    *(Kết quả là một đường cong parabol)*

*   **Mô hình đa thức bậc 3:**
    $$y = w_1x + w_2x^2 + w_3x^3 + b$$
    *(Kết quả là đường cong có thể uốn lượn nhiều hơn)*

---

### Đặc điểm quan trọng
*   **Vẫn là mô hình tuyến tính:** Dù đường dự đoán là đường cong, nhưng về mặt toán học, mô hình vẫn là **tuyến tính đối với các trọng số ($w$)**. Điều này có nghĩa là bạn vẫn có thể dùng các thuật toán tối ưu đơn giản như Gradient Descent để huấn luyện.
*   **Tương tác đặc trưng (Feature Crosses):** Ngoài việc nâng lũy thừa, bạn có thể nhân hai đặc trưng khác nhau (ví dụ: $x_1 \times x_2$). Đây cũng được coi là một dạng biến đổi đa thức giúp mô hình học được sự kết hợp giữa các yếu tố.

### Nguy cơ Overfitting (Quá khớp)
Đây là lưu ý quan trọng nhất khi dùng đa thức:
*   **Bậc đa thức càng cao:** Mô hình càng linh hoạt và có thể khớp cực chính xác với mọi điểm dữ liệu trong tập huấn luyện.
*   **Hậu quả:** Mô hình sẽ bắt đầu học cả những "nhiễu" (noise) trong dữ liệu thay vì học quy luật chung. Khi đưa dữ liệu mới vào, mô hình sẽ dự đoán rất sai.
*   **Quy tắc:** Luôn bắt đầu với bậc thấp (bậc 2 hoặc 3) và sử dụng **Regularization** để kiểm soát các trọng số.

---

### Ghi nhớ cho ML:
1.  Dùng **Polynomial Transforms** khi bạn thấy biểu đồ dữ liệu có dạng **đường cong**.
2.  Nó biến mô hình tuyến tính thành một bộ phân loại/dự báo mạnh mẽ hơn mà không cần đổi sang các thuật toán phức tạp khác.
3.  **Cảnh giác với bậc cao:** Đừng lạm dụng lũy thừa quá lớn vì sẽ gây lãng phí tài nguyên và dẫn đến Overfitting.
