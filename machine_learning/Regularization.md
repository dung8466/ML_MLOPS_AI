
# Điều quy hóa (Regularization) từ con số 0

Regularization là một kỹ thuật cực kỳ quan trọng trong Machine Learning được sử dụng để ngăn chặn hiện tượng **Overfitting** (Quá khớp). Nó giúp mô hình không quá phụ thuộc vào các điểm dữ liệu nhiễu và tăng khả năng dự báo trên dữ liệu mới.

## 1. Khái niệm cơ bản

Mục tiêu của Regularization là làm cho mô hình trở nên "đơn giản" hơn bằng cách thêm một đại lượng phạt (penalty) vào hàm mất mát (Cost Function).

**Cơ chế hoạt động:**
1. **Phạt trọng số:** Nếu các trọng số ($w$ hoặc $\theta$) quá lớn, giá trị phạt sẽ tăng vọt.
2. **Tối ưu hóa:** Thuật toán buộc phải tìm bộ trọng số vừa làm giảm sai số dự báo, vừa phải có giá trị nhỏ để giảm tiền phạt.
3. **Kết quả:** Mô hình học được các quy luật tổng quát thay vì học thuộc lòng từng điểm dữ liệu huấn luyện.

---

## 2. Các loại Regularization phổ biến

Có hai kỹ thuật chính dựa trên cách tính toán hình phạt:

**L2 Regularization (Ridge Regression):**
Phạt dựa trên **tổng bình phương** của các trọng số. Nó ép các trọng số nhỏ đi nhưng hiếm khi triệt tiêu chúng về 0.
$$J(\theta) = MSE + \lambda \sum_{j=1}^{n} \theta_j^2$$

**L1 Regularization (Lasso Regression):**
Phạt dựa trên **tổng giá trị tuyệt đối** của các trọng số. Nó có khả năng ép các trọng số không quan trọng về đúng bằng 0 (giúp chọn lọc đặc trưng).
$$J(\theta) = MSE + \lambda \sum_{j=1}^{n} |\theta_j|$$

*Trong đó $\lambda$ (Lambda) là siêu tham số điều khiển mức độ phạt.*

---

## 3. Triển khai L2 Regularization bằng Python

Dưới đây là cách tích hợp L2 vào hàm tính toán Loss và Gradient trong Hồi quy tuyến tính:

```python
import numpy as np

def compute_cost_reg(X, y, theta, lamb):
    m = len(y)
    # Tính sai số dự báo (MSE)
    h = X.dot(theta)
    loss = (1 / (2 * m)) * np.sum(np.square(h - y))
    
    # Tính phần phạt Regularization (không phạt trọng số chặn theta[0])
    reg_penalty = (lamb / (2 * m)) * np.sum(np.square(theta[1:]))
    
    return loss + reg_penalty

def gradient_descent_reg(X, y, theta, lr, lamb, iterations):
    m = len(y)
    for _ in range(iterations):
        h = X.dot(theta)
        gradient = (1 / m) * (X.T.dot(h - y))
        
        # Cập nhật Gradient với phần phạt (không phạt theta[0])
        reg_term = (lamb / m) * theta
        reg_term[0] = 0 # Loại bỏ phạt cho bias
        
        theta = theta - lr * (gradient + reg_term)
    return theta
```

---

## 4. Tác động của siêu tham số $\lambda$ (Lambda)

Việc chọn $\lambda$ là sự đánh đổi giữa **Bias** (Độ chệch) và **Variance** (Phương sai):

- **$\lambda = 0$:** Không có regularization. Mô hình dễ bị **Overfitting** (High Variance).
- **$\lambda$ quá lớn:** Các trọng số bị ép quá mạnh về gần 0. Mô hình trở nên quá đơn giản, không học được gì, dẫn đến **Underfitting** (High Bias).
- **$\lambda$ tối ưu:** Cân bằng được sai số huấn luyện và khả năng tổng quát hóa. Thường được chọn thông qua tập **Cross Validation**.

---

## 5. So sánh L1 vs L2

| Đặc điểm | L2 Regularization (Ridge) | L1 Regularization (Lasso) |
| :--- | :--- | :--- |
| **Hình phạt** | Bình phương trọng số ($\theta^2$) | Trị tuyệt đối trọng số ($|\theta|$) |
| **Trọng số** | Giảm dần về gần 0 | Có thể bằng đúng 0 |
| **Sử dụng khi** | Hầu hết các đặc trưng đều có ích | Chỉ có một ít đặc trưng là quan trọng |
| **Ưu điểm** | Tính toán đạo hàm dễ dàng | Giúp loại bỏ đặc trưng dư thừa |

---

## 6. Những nội dung cần lưu ý

1. **Chuẩn hóa dữ liệu (Feature Scaling):** Đây là bước **bắt buộc**. Vì Regularization phạt dựa trên độ lớn của trọng số, nếu các đặc trưng không cùng quy mô, mô hình sẽ phạt sai các đặc trưng có giá trị lớn.
2. **Không phạt Bias:** Thông thường chúng ta không áp dụng regularization cho trọng số chặn ($\theta_0$ hoặc bias) vì nó không gây ra sự phức tạp quá mức cho đường cong.
3. **Đồ thị Loss:** Khi dùng Regularization, đường cong Loss trên tập huấn luyện có thể cao hơn một chút so với khi không dùng, nhưng Loss trên tập Validation sẽ ổn định và thấp hơn.

---
