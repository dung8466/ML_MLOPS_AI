# Linear regression

```

y` = b + w1x1 + w2x2 + w3x3 + w4x4 + w5x5

```
 Trong đó 
 + y': predicted label
 + b: bias -> tính toán từ traing
 + w: weight -> tính toán từ traing
 + x: feature value

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

### Cách chọn Loss 

Khi chọn loss, cân nhắc cách bạn muốn mô hình xử lý các giá trị ngoại lệ.
MSE làm cho mô hình dịch chuyển nhiều hơn về phía các giá trị ngoại lệ, trong khi MAE thì không. L2 gây ra mức phạt cao hơn nhiều đối với một giá trị ngoại lệ so với mất mát L1.

MAE model:

![MAE model](images/model-mae.png "MAE")

MSE model:

![MSE model](images/model-mse.png "MSE")

## Gradient descent

Kỹ thuật toán học tìm kiếm lặp đi lặp lại các trọng số và độ lệch sao cho tạo ra mô hình có loss thấp nhất. Gradient descent tìm ra trọng số và độ lệch tốt nhất bằng cách lặp lại quy trình sau đây với số lần lặp do người dùng xác định.

+ B1: Tính toán loss với weight hiện tại và bias
+ B2: Xác định hướng di chuyển của các trọng số và độ lệch để giảm thiểu loss
+ B3: Điều chỉnh giá trị trọng số và độ lệch một chút theo hướng giảm loss
+ B4: Quay lại bước một và lặp lại quy trình cho đến khi mô hình không thể giảm thiểu loss thêm nữa

![gradient descent](images/gradient-descent.png "gradient descent")

### Công thức

#### 1. Hàm dự đoán (Prediction Function)

Với một đặc trưng (feature) `x`, trọng số `w`, và độ chệch `b`:

$$ f_{w,b}(x) = (w \times x) + b $$

Trong đó:
- `x`: giá trị đặc trưng đầu vào (ví dụ: trọng lượng xe - nghìn pounds)
- `w`: trọng số (weight) - hệ số góc
- `b`: độ chệch (bias) - điểm chắn trục tung
- `f_{w,b}(x)`: giá trị dự đoán (ví dụ: miles per gallon)

#### 2. Hàm mất mát - MSE (Mean Squared Error)

Với tập dữ liệu có `M` mẫu:

$$ \text{Loss}_{MSE} = \frac{1}{M} \sum_{i=1}^{M} (f_{w,b}(x^{(i)}) - y^{(i)})^2 $$

Trong đó:
- `M`: tổng số mẫu trong tập dữ liệu
- `i`: chỉ số của mẫu thứ i
- `x^{(i)}`: giá trị đặc trưng của mẫu thứ i
- `y^{(i)}`: giá trị thực tế (label) của mẫu thứ i
- `f_{w,b}(x^{(i)})`: giá trị dự đoán cho mẫu thứ i

#### 3. Gradient (Đạo hàm) của hàm MSE

##### 3.1. Đạo hàm theo trọng số (Weight derivative)

$$ \frac{\partial}{\partial w} \left( \frac{1}{M} \sum_{i=1}^{M} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \right) = \frac{1}{M} \sum_{i=1}^{M} (f_{w,b}(x^{(i)}) - y^{(i)}) \times 2x^{(i)} $$

**Cách tính từng bước:**
1. Với mỗi mẫu i: tính `(giá trị dự đoán - giá trị thực) * 2 * (giá trị đặc trưng x)`
2. Cộng tất cả các giá trị này lại
3. Chia tổng cho số lượng mẫu `M`

**Ký hiệu rút gọn:**

$$ \nabla w = \frac{\partial \text{MSE}}{\partial w} = \frac{2}{M} \sum_{i=1}^{M} (f_{w,b}(x^{(i)}) - y^{(i)}) \cdot x^{(i)} $$

##### 3.2. Đạo hàm theo độ chệch (Bias derivative)

$$ \frac{\partial}{\partial b} \left( \frac{1}{M} \sum_{i=1}^{M} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \right) = \frac{1}{M} \sum_{i=1}^{M} (f_{w,b}(x^{(i)}) - y^{(i)}) \times 2 $$

**Cách tính từng bước:**
1. Với mỗi mẫu i: tính `(giá trị dự đoán - giá trị thực) * 2`
2. Cộng tất cả các giá trị này lại
3. Chia tổng cho số lượng mẫu `M`

**Ký hiệu rút gọn:**

$$ \nabla b = \frac{\partial \text{MSE}}{\partial b} = \frac{2}{M} \sum_{i=1}^{M} (f_{w,b}(x^{(i)}) - y^{(i)}) $$

**Chú ý:** Không nhân với `x` vì đạo hàm của `b` đối với `b` là 1.

#### 4. Cập nhật tham số (Parameter Update)

Sau khi tính được gradient, ta cập nhật trọng số và độ chệch bằng cách **đi ngược hướng gradient**:

##### 4.1. Cập nhật trọng số

$$ w_{\text{mới}} = w_{\text{cũ}} - (\text{learning rate} \times \nabla w) $$

Hay viết đầy đủ:

$$ w_{\text{mới}} = w_{\text{cũ}} - \eta \times \frac{2}{M} \sum_{i=1}^{M} (f_{w,b}(x^{(i)}) - y^{(i)}) \cdot x^{(i)} $$

##### 4.2. Cập nhật độ chệch

$$ b_{\text{mới}} = b_{\text{cũ}} - (\text{learning rate} \times \nabla b) $$

Hay viết đầy đủ:

$$ b_{\text{mới}} = b_{\text{cũ}} - \eta \times \frac{2}{M} \sum_{i=1}^{M} (f_{w,b}(x^{(i)}) - y^{(i)}) $$

Trong đó:
- `η` (eta): **learning rate** (tốc độ học) - là một số dương nhỏ (ví dụ: 0.01)
- Dấu **trừ** (`-`) thể hiện việc di chuyển **ngược hướng** gradient


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

#### 6. Tóm tắt các công thức chính

| Thành phần | Công thức |
|-----------|-----------|
| Dự đoán | \( \hat{y} = w \cdot x + b \) |
| MSE Loss | \( \frac{1}{M} \sum (\hat{y}_i - y_i)^2 \) |
| Gradient theo w | \( \frac{2}{M} \sum (\hat{y}_i - y_i) \cdot x_i \) |
| Gradient theo b | \( \frac{2}{M} \sum (\hat{y}_i - y_i) \) |
| Cập nhật w | \( w = w - \eta \cdot \nabla w \) |
| Cập nhật b | \( b = b - \eta \cdot \nabla b \) |

---

**Lưu ý quan trọng:** Các công thức trên áp dụng cho **batch gradient descent** (tính gradient trên toàn bộ dữ liệu). Với SGD hoặc mini-batch, chỉ khác ở số lượng mẫu `M` dùng để tính.


