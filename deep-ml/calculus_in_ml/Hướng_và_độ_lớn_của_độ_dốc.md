# Vector Gradient: Magnitude và Direction

Giả sử ta có vector Gradient $\mathbf{g}$ của hàm số $f$ tại điểm $\mathbf{w}$:

$$
\mathbf{g} = \nabla f(\mathbf{w}) = \begin{bmatrix} \frac{\partial f}{\partial w_1} \\ \frac{\partial f}{\partial w_2} \\ \vdots \\ \frac{\partial f}{\partial w_n} \end{bmatrix}
$$

---

### 1. Magnitude (Độ lớn - L2 Norm)
Đại diện cho **độ dốc** (steepness) của hàm số. Magnitude càng lớn, hàm số thay đổi càng nhanh tại điểm đó.

**Công thức:**

$$
\|\mathbf{g}\|_2 = \sqrt{\sum_{i=1}^n g_i^2}
$$

---

### 2. Direction of Steepest Ascent (Hướng tăng nhanh nhất)
Là một **vector đơn vị** (unit vector) chỉ về hướng mà giá trị của hàm số tăng lên nhanh nhất. Trong Machine Learning, đây là hướng đi ngược lại với mục tiêu của tối ưu hóa.

**Công thức:**

$$
\mathbf{u}_{\text{ascent}} = \frac{\mathbf{g}}{\|\mathbf{g}\|_2}
$$

*(Điều kiện: $\|\mathbf{g}\|_2 \neq 0$)*

---

### 3. Descent Direction (Hướng giảm nhanh nhất - Steepest Descent)
Là hướng mà giá trị hàm số giảm xuống nhanh nhất. Đây là hướng cốt lõi được sử dụng trong thuật toán **Gradient Descent** để cập nhật trọng số.

**Công thức:**

$$
\mathbf{u}_{\text{descent}} = -\frac{\mathbf{g}}{\|\mathbf{g}\|_2}
$$

---

### 4. Quy tắc cập nhật trọng số trong ML
Trong thuật toán Gradient Descent, chúng ta kết hợp cả hướng, độ lớn và tốc độ học ($\eta$):

$$
\mathbf{w}_{\text{new}} = \mathbf{w}_{\text{old}} - \eta \cdot \mathbf{g}
$$

Nếu phân tích theo **Direction** và **Magnitude**:

$$
\mathbf{w}_{\text{new}} = \mathbf{w}_{\text{old}} + \eta \cdot \|\mathbf{g}\|_2 \cdot \mathbf{u}_{\text{descent}}
$$

---

### Trường hợp đặc biệt: Zero Vector
Nếu $\mathbf{g} = \mathbf{0}$ (tại điểm cực trị):
*   **Magnitude:** $\|\mathbf{g}\|_2 = 0$
*   **Direction:** Trả về vector $\mathbf{0}$.
*   **Ý nghĩa:** Mô hình đã hội tụ.

---


## Code Implementation

```python
import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
    # Chuyển đổi list đầu vào thành numpy array để tính toán vector
    grad_arr = np.array(gradient, dtype=float)
    
    # 1. Tính Magnitude (Độ lớn - L2 norm)
    # Công thức: sqrt(sum(xi^2))
    magnitude = np.linalg.norm(grad_arr)
    
    # 2. Xử lý trường hợp đặc biệt: Vector không (điểm tới hạn)
    if magnitude == 0:
        # Nếu magnitude bằng 0, hướng và hướng đi xuống đều là vector không
        direction = np.zeros_like(grad_arr).tolist()
        descent_direction = np.zeros_like(grad_arr).tolist()
    else:
        # 3. Tính hướng (Unit Vector - Steepest Ascent)
        # Công thức: v / ||v||
        direction_arr = grad_arr / magnitude
        direction = direction_arr.tolist()
        
        # 4. Tính hướng đi xuống (Steepest Descent)
        # Công thức: -hướng
        descent_direction = (-direction_arr).tolist()
    
    # Trả về kết quả dưới dạng dictionary
    return {
        'magnitude': float(magnitude),
        'direction': direction,
        'descent_direction': descent_direction
    }
```
