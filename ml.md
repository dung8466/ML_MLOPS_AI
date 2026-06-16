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

Dưới đây là bản dịch và trình bày nội dung về **Lựa chọn chỉ số và sự đánh đổi (Tradeoffs)** sang tiếng Việt dưới dạng Markdown:

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
