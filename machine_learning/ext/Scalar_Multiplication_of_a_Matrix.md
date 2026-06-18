# Lý thuyết

---

# Phép nhân Ma trận với một số vô hướng (Scalar Multiplication of a Matrix)

Khi một ma trận $A$ được nhân với một số vô hướng $k$, phép toán này được định nghĩa là nhân **mỗi phần tử** của $A$ với $k$.

Cho một ma trận $A$:

$$
A = 
\begin{pmatrix} 
a_{11} & a_{12} \\ 
a_{21} & a_{22} 
\end{pmatrix}
$$

Và một số vô hướng $k$, kết quả của phép nhân $kA$ là:

$$
kA = 
\begin{pmatrix} 
k a_{11} & k a_{12} \\ 
k a_{21} & k a_{22} 
\end{pmatrix}
$$

### Đặc điểm:
Thao tác này làm thay đổi tỷ lệ (scale) của ma trận theo hệ số $k$ mà không làm thay đổi kích thước (số hàng, số cột) hoặc tỷ lệ tương đối giữa các phần tử bên trong ma trận đó.

---

**💡 Mẹo cho ML:** Trong lập trình với **NumPy**, bạn chỉ cần dùng toán tử `*` (ví dụ: `k * A`), thư viện sẽ tự động thực hiện phép nhân này trên toàn bộ các phần tử của mảng (đây gọi là cơ chế *broadcasting*).

---

# Bài tập

Write a Python function that multiplies a matrix by a scalar and returns the result.
Example:
Input:

    matrix = [[1, 2], [3, 4]], scalar = 2

Output:

    [[2, 4], [6, 8]]

Reasoning:

    Each element of the matrix is multiplied by the scalar.

# Code

```python
def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	# Your code here
  result = [[element * scalar for element in row] for row in matrix]
  return result
```
