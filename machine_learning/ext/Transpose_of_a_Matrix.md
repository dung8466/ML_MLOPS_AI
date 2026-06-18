# Lý thuyết

---

# Chuyển vị của một Ma trận (Transpose of a Matrix)

Ma trận chuyển vị được tạo ra bằng cách hoán đổi hàng và cột của nó. Nếu $A$ là một ma trận kích thước $m \times n$, thì ma trận chuyển vị của nó, ký hiệu là $A^T$, là một ma trận kích thước $n \times m$.

### Định nghĩa

Đối với một ma trận $A$, trong đó phần tử $A_{ij}$ nằm ở hàng $i$ và cột $j$:

$$
(A^T)_{ij} = A_{ji}
$$

### Ví dụ

**Ma trận gốc $A$ ($2 \times 3$):**

$$
A = 
\begin{pmatrix} 
1 & 2 & 3 \\ 
4 & 5 & 6 
\end{pmatrix}
$$

**Ma trận chuyển vị $A^T$ ($3 \times 2$):**

$$
A^T = 
\begin{pmatrix} 
1 & 4 \\ 
2 & 5 \\ 
3 & 6 
\end{pmatrix}
$$

---

### Các tính chất của Ma trận chuyển vị

1.  **Chuyển vị kép:** $(A^T)^T = A$
2.  **Phép cộng:** $(A + B)^T = A^T + B^T$
3.  **Nhân với số vô hướng:** $(cA)^T = cA^T$
4.  **Phép nhân (tích):** $(AB)^T = B^T A^T$ (Lưu ý: thứ tự bị đảo ngược)
5.  **Ma trận đối xứng:** Nếu $A = A^T$ thì $A$ được gọi là ma trận đối xứng.

---

### Cài đặt trong Python

Sử dụng hàm `zip(*matrix)` là một cách viết đặc trưng (Pythonic) để thực hiện chuyển vị:

*   `*matrix`: Giải nén (unpack) các hàng thành các đối số riêng biệt.
*   `zip()`: Nhóm các phần tử theo vị trí tương ứng (phần tử đầu tiên đi với nhau, phần tử thứ hai đi với nhau, v.v.).
*   Chuyển đổi từng tuple kết quả thành một danh sách (list) để có kết quả cuối cùng.

---

### Ứng dụng

*   **Đại số tuyến tính:** Giải hệ phương trình tuyến tính.
*   **Machine Learning:** Thay đổi hình dạng (reshaping) ma trận đặc trưng, tính toán ma trận hiệp phương sai.
*   **Đồ họa máy tính:** Các ma trận biến đổi không gian.
*   **Xử lý dữ liệu:** Chuyển đổi giữa các định dạng lưu trữ theo hàng (row-major) và theo cột (column-major).

---

# Bài tập

Write a Python function that computes the transpose of a given 2D matrix. The transpose of a matrix is formed by turning its rows into columns and columns into rows. For an m×n matrix, the transposed matrix will have dimensions n×m.

Example:

Input:

    a = [[1, 2, 3], [4, 5, 6]]

Output:

    [[1, 4], [2, 5], [3, 6]]

Reasoning:

The input is a 2×3 matrix. The transpose swaps rows and columns: the first row [1, 2, 3] becomes the first column, and the second row [4, 5, 6] becomes the second column, resulting in a 3×2 matrix.

# Code

```python
def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    # Your code here
    b = [list(row) for row in zip(*a)]
    return b
```

Dùng numpy

```python
import numpy as np

def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    # 1. Chuyển list thông thường thành numpy array
    arr = np.array(a)
    
    # 2. Sử dụng thuộc tính .T để chuyển vị
    # Hoặc dùng np.transpose(arr)
    transposed = arr.T
    
    # 3. Chuyển ngược từ numpy array về lại list of lists
    return transposed.tolist()
```
