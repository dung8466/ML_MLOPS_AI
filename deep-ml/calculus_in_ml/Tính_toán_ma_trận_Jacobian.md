### Hiểu Toán trước: Ma trận Jacobian là gì?

Nếu chúng ta có một hàm vector $\mathbf{f}$ nhận đầu vào là một vector $n$ chiều và trả về một vector $m$ chiều:
$$\mathbf{f}(\mathbf{x}) = [f_1(x_1, \dots, x_n), f_2(x_1, \dots, x_n), \dots, f_m(x_1, \dots, x_n)]^T$$

**Ma trận Jacobian** ($J$) là ma trận chứa tất cả các đạo hàm riêng bậc nhất của hàm số đó:

$$J = \begin{bmatrix} 
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \dots & \frac{\partial f_1}{\partial x_n} \\ 
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \dots & \frac{\partial f_2}{\partial x_n} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \dots & \frac{\partial f_m}{\partial x_n} 
\end{bmatrix}$$

Để tính toán **Ma trận Jacobian** bằng phương pháp đạo hàm số (numerical differentiation), chúng ta sẽ sử dụng công thức **Forward Difference** (Sai phân tiến).

---

### Triển khai Code với NumPy

```python
import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
    """
    Tính ma trận Jacobian bằng phương pháp sai phân hữu hạn.
    """
    x_arr = np.array(x, dtype=float)
    n = len(x)
    
    # 1. Tính giá trị hàm số tại điểm gốc f(x)
    # Giả sử f(x) trả về một list hoặc array độ dài m
    f_x = np.array(f(x_arr.tolist()))
    m = len(f_x)
    
    # 2. Khởi tạo ma trận Jacobian rỗng kích thước (m x n)
    jacobian = np.zeros((m, n))
    
    # 3. Tính đạo hàm riêng cho từng biến đầu vào (từng cột của Jacobian)
    for j in range(n):
        # Tạo một bản sao của x và cộng thêm một lượng cực nhỏ h vào biến thứ j
        x_perturbed = x_arr.copy()
        x_perturbed[j] += h
        
        # Tính f(x + h)
        f_x_plus_h = np.array(f(x_perturbed.tolist()))
        
        # Công thức đạo hàm số: [f(x + h) - f(x)] / h
        # Đây là cột thứ j của ma trận Jacobian
        jacobian[:, j] = (f_x_plus_h - f_x) / h
        
    return jacobian.tolist()
```

---

### Giải thích logic lập trình:

1.  **Đạo hàm số (Numerical Differentiation):** Vì chúng ta không có công thức giải tích của $f$, chúng ta sử dụng định nghĩa đạo hàm: $f'(x) \approx \frac{f(x+h) - f(x)}{h}$ với $h$ rất nhỏ.
2.  **Tính theo cột:** Trong vòng lặp `for j in range(n)`, chúng ta làm thay đổi biến thứ $j$ của đầu vào. Kết quả nhận được chính là đạo hàm của **tất cả các đầu ra** đối với **biến đầu vào đó**. Đây chính là cột thứ $j$ của ma trận Jacobian.
3.  **Slicing `jacobian[:, j]`:** Đây là kỹ thuật của NumPy để gán giá trị cho toàn bộ một cột trong ma trận một cách nhanh chóng.

### Ứng dụng trong Machine Learning:
*   **Neural Networks:** Khi tính toán Gradient cho các lớp có đầu ra là vector (như lớp Softmax), ma trận Jacobian được dùng để truyền ngược sai số.
*   **Robot học (Robotics):** Dùng để tính toán mối quan hệ giữa tốc độ của các khớp nối và tốc độ của bàn tay robot.
*   **Tối ưu hóa:** Là thành phần cốt lõi trong các thuật toán tối ưu bậc cao như **Levenberg-Marquardt**.
