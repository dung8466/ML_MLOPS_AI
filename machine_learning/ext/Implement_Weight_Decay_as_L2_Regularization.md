# Lý thuyết

---

## Tìm hiểu về Suy giảm Trọng số (Weight Decay) và Điều quy hóa L2

Suy giảm trọng số (Weight decay) là một kỹ thuật điều quy hóa được sử dụng để ngăn chặn hiện tượng quá khớp (overfitting) trong mạng nơ-ron bằng cách xử phạt các trọng số có giá trị lớn. Nó có mối quan hệ chặt chẽ với điều quy hóa L2, mặc dù cách cài đặt có đôi chút khác biệt.

### 1. Vấn đề của việc không Điều quy hóa

Nếu không có điều quy hóa, mạng nơ-ron có thể phát triển các trọng số rất lớn dẫn đến:
*   **Overfitting (Quá khớp):** Mô hình ghi nhớ dữ liệu huấn luyện thay vì học các quy luật có thể khái quát hóa.
*   **Khả năng khái quát hóa kém:** Độ chính xác trên tập huấn luyện cao nhưng trên tập kiểm tra thấp.
*   **Mất ổn định số học:** Trọng số lớn có thể gây ra hiện tượng bùng nổ gradient (gradient explosion).

### 2. Điều quy hóa L2 trong Hàm mất mát

Điều quy hóa L2 thêm một đại lượng phạt vào hàm mất mát:
$$L_{total} = L_{original} + \frac{\lambda}{2} \sum_{i} w_i^2$$

**Trong đó:**
*   $L_{original}$: Hàm mất mát gốc (ví dụ: cross-entropy).
*   $\lambda$: Cường độ điều quy hóa (hệ số suy giảm trọng số).
*   $w_i$: Các trọng số của mô hình.
*   Hệ số $1/2$: Được thêm vào để triệt tiêu số mũ khi tính đạo hàm, giúp công thức gọn hơn.

### 3. Quy tắc cập nhật Suy giảm Trọng số

Khi tính gradient của hàm mất mát đã được điều quy hóa và áp dụng Gradient Descent, ta có:
$$w_{new} = w - \eta \frac{\partial L_{total}}{\partial w}$$

Khai triển gradient:
$$\frac{\partial L_{total}}{\partial w} = \frac{\partial L_{original}}{\partial w} + \lambda w$$

Thay ngược trở lại:
$$w_{new} = w - \eta \left( \frac{\partial L_{original}}{\partial w} + \lambda w \right)$$

Rút gọn thành:
$$w_{new} = w - \eta \cdot g - \eta \cdot \lambda \cdot w$$

Trong đó $g = \frac{\partial L_{original}}{\partial w}$ là gradient gốc. Công thức này có thể viết lại là:
$$w_{new} = (1 - \eta\lambda)w - \eta \cdot g$$

Biểu thức $(1 - \eta\lambda)$ làm cho các trọng số "suy giảm" dần về 0 sau mỗi bước, đó là lý do kỹ thuật này có tên là **"Weight Decay"**.

---

### 4. Tại sao loại bỏ Bias (Độ chệch)?

Theo quy ước, suy giảm trọng số thường không áp dụng cho các tham số bias. Lý do là:
1.  **Bias không gây quá khớp:** Bias chỉ dịch chuyển hàm kích hoạt chứ không khuếch đại tín hiệu đầu vào như trọng số.
2.  **Thực nghiệm:** Việc loại bỏ bias khỏi quá trình điều quy hóa cho kết quả tốt hơn.
3.  **Lý thuyết:** Điều quy hóa L2 phạt độ phức tạp; bias không làm tăng độ phức tạp của mô hình nhiều như trọng số.

---

### 5. Ví dụ tính toán

**Giả sử:**
*   Trọng số hiện tại: $w = 1.0$
*   Gradient: $g = 0.1$
*   Tốc độ học: $\eta = 0.1$
*   Hệ số suy giảm: $\lambda = 0.01$

**Các bước tính:**
1.  Thành phần Gradient Descent: $\eta \cdot g = 0.1 \times 0.1 = 0.01$
2.  Thành phần Suy giảm trọng số: $\eta \cdot \lambda \cdot w = 0.1 \times 0.01 \times 1.0 = 0.001$
3.  Trọng số mới: $w_{new} = 1.0 - 0.01 - 0.001 = 0.989$

---

### 6. So sánh L2 Regularization vs Weight Decay

Dù thường được dùng thay thế cho nhau, giữa chúng có sự khác biệt tinh tế:

| Khía cạnh | Điều quy hóa L2 | Suy giảm trọng số (Weight Decay) |
| :--- | :--- | :--- |
| **Cài đặt** | Thêm hình phạt vào hàm loss | Trực tiếp sửa đổi bước cập nhật trọng số |
| **Với Momentum** | Hành vi khác biệt | Đơn giản và hiệu quả hơn |
| **Với Adam (Tốc độ học thích ứng)** | Không tương đương | Thường được ưu tiên hơn (AdamW) |
| **Tính tương đồng toán học** | Chỉ đúng với SGD tiêu chuẩn | Là cách cài đặt trực tiếp |

---

### 7. Mẹo thực hành

*   **Giá trị thông dụng:** Bắt đầu với $\lambda = 0.0001$ hoặc $0.01$.
*   **Tinh chỉnh siêu tham số:** Đây là một thông số cực kỳ quan trọng cần được thử nghiệm nhiều lần.
*   **Theo dõi huấn luyện:** Suy giảm quá mạnh $\rightarrow$ Underfitting (Dưới khớp); Suy giảm quá nhẹ $\rightarrow$ Overfitting.

### 8. Cài đặt trong các thư viện phổ biến

*   **PyTorch:** `torch.optim.SGD(params, lr=0.1, weight_decay=0.01)`
*   **TensorFlow:** `tf.keras.optimizers.SGD(learning_rate=0.1, weight_decay=0.01)`
*(Cả hai thư viện đều hỗ trợ tự động loại bỏ bias nếu được cấu hình đúng).*

---
**Kết luận:** Suy giảm trọng số là một trong những kỹ thuật điều quy hóa hiệu quả và được sử dụng rộng rãi nhất trong Deep Learning, giúp mô hình khái quát hóa tốt hơn trong khi vẫn giữ cách cài đặt đơn giản.

# Bài tập

Implement weight decay (L2 regularization) for neural network parameters. Given parameter arrays, their gradients, a learning rate, and a weight decay factor, apply the parameter update that includes both gradient descent and L2 regularization. The function should take a boolean list indicating which parameter groups should have weight decay applied (typically, weight decay is applied to weights but not to biases). Return the updated parameters after applying the appropriate update rule.
Example:
Input:

    parameters=[[1.0, 2.0]], gradients=[[0.1, 0.2]], lr=0.1, weight_decay=0.01, apply_to_all=[True]

Output:

    [[0.989, 1.978]]

Reasoning:

    For the first parameter (1.0): 1.0−0.1×0.1−0.1×0.01×1.0=1.0−0.01−0.001=0.9891.0−0.1×0.1−0.1×0.01×1.0=1.0−0.01−0.001=0.989. For the second parameter (2.0): 2.0−0.1×0.2−0.1×0.01×2.0=2.0−0.02−0.002=1.9782.0−0.1×0.2−0.1×0.01×2.0=2.0−0.02−0.002=1.978.

# Code

```python
def apply_weight_decay(parameters: list[list[float]], gradients: list[list[float]], 
                       lr: float, weight_decay: float, apply_to_all: list[bool]) -> list[list[float]]:
	# Your code here
    updated_params = []
    
    for param_array, grad_array, apply_wd in zip(parameters, gradients, apply_to_all):
        updated_array = []
        for param, grad in zip(param_array, grad_array):
            if apply_wd:
                updated_param = param - lr * grad - lr * weight_decay * param
            else:
                updated_param = param - lr * grad
            updated_array.append(updated_param)
        updated_params.append(updated_array)
    
    return updated_params
```
