# Lý thuyết

---

# Tìm hiểu về Quy tắc Cramer (Understanding Cramer's Rule)

Quy tắc Cramer là một phương pháp dùng để giải hệ phương trình tuyến tính dạng $Ax = b$ bằng cách sử dụng định thức (determinants).

### Điều kiện áp dụng
*   Ma trận hệ số $A$ phải là **ma trận vuông** ($n \times n$).
*   Định thức của $A$, ký hiệu là $\det(A)$, phải **khác không** để hệ có nghiệm duy nhất.

### Công thức
Đối với mỗi biến $x_i$, ta thay thế cột thứ $i$ của ma trận $A$ bằng vector kết quả $b$ và tính toán:

$$x_i = \frac{\det(A_i)}{\det(A)}$$

**Trong đó:**
*   $A_i$ là ma trận được hình thành bằng cách thay thế cột thứ $i$ của $A$ bằng vector $b$.
*   $\det(A)$ là định thức của ma trận gốc $A$.

---

### Các bước thực hiện
1.  Tính $\det(A)$. Nếu định thức bằng 0, hệ phương trình không có nghiệm duy nhất (trả về -1 hoặc thông báo lỗi).
2.  Đối với mỗi biến $x_i$:
    *   Thay thế cột $i$ trong ma trận $A$ bằng vector $b$.
    *   Tính định thức của ma trận mới thu được, ký hiệu là $\det(A_i)$.
    *   Tính giá trị của biến: $x_i = \frac{\det(A_i)}{\det(A)}$.

---

### Ví dụ minh họa
Cho hệ phương trình:

$$
A = 
\begin{bmatrix} 
2 & -1 & 3 \\ 
4 & 2 & 1 \\ 
-6 & 1 & -2 
\end{bmatrix}
$$

$$
b = 
\begin{bmatrix} 
5 \\ 
10 \\ 
-3 
\end{bmatrix}
$$

1.  Tính định thức ma trận gốc: $\det(A) = -36.0$.
2.  Thay thế từng cột bằng vector $b$ để tính các định thức phụ:
    *   $\det(A_1) = -6.0$
    *   $\det(A_2) = -120.0$
    *   $\det(A_3) = -96.0$

**Kết quả:**
$$x = \left[ \frac{-6}{-36}, \frac{-120}{-36}, \frac{-96}{-36} \right] = [0.1667, 3.3333, 2.6667]$$

---

### Ứng dụng
*   Giải các hệ phương trình tuyến tính quy mô nhỏ.
*   Hữu ích trong các bài toán đại số tuyến tính lý thuyết.
*   **Hạn chế:** Không thực tế khi áp dụng cho các ma trận lớn vì chi phí tính toán định thức cực kỳ tốn kém.

---
**💡 Mẹo cho ML:** Mặc dù quy tắc Cramer rất thú vị về mặt toán học, nhưng trong Machine Learning thực tế, chúng ta thường sử dụng các phương pháp như **Phân rã LU** hoặc **Gradient Descent** để giải hệ phương trình vì chúng hiệu quả hơn nhiều về mặt tốc độ xử lý.

# Bài tập

Implement a function to solve a system of linear equations Ax=bAx=b using Cramer's Rule. The function should take a square coefficient matrix AA and a constant vector bb, and return the solution vector xx. If the system has no unique solution (i.e., the determinant of AA is zero), return -1.
Example:
Input:

    A = [[2, -1, 3], [4, 2, 1], [-6, 1, -2]], b = [5, 10, -3]

Output:

    [0.1667 3.3333 2.6667]

Reasoning:

    We compute the determinant of A and then replace each column with vector b to compute the determinants of modified matrices. These are then used in the formula xi=det⁡(Ai)det⁡(A)xi​=det(A)det(Ai​)​ to get the solution.

# Code

```python
import numpy as np

def cramers_rule(A, b):
    # Chuyển đổi đầu vào thành numpy array để xử lý toán học
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    # 1. Tính định thức của ma trận gốc A
    det_A = np.linalg.det(A)
    
    # 2. Kiểm tra nếu định thức bằng 0 (hệ không có nghiệm duy nhất)
    # Sử dụng np.isclose để tránh lỗi sai số dấu phẩy động
    if np.isclose(det_A, 0):
        return -1
    
    # Sử dụng pass như một lệnh giữ chỗ nếu cần, nhưng ở đây ta đi thẳng vào logic
    pass 

    n = len(b)
    x = []
    
    # 3. Tính toán từng biến x_i
    for i in range(n):
        # Tạo một bản sao của ma trận A để không làm thay đổi ma trận gốc
        Ai = A.copy()
        
        # Thay thế cột thứ i bằng vector b
        Ai[:, i] = b
        
        # Tính định thức của ma trận mới Ai
        det_Ai = np.linalg.det(Ai)
        
        # Tính giá trị biến x_i = det(Ai) / det(A)
        x.append(det_Ai / det_A)
        
    return x
```
