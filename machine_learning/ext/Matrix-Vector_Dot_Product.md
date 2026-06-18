# Lý thuyết

---

# Tích vô hướng Ma trận - Vector (Matrix-Vector Dot Product)

Xét một ma trận $A$ và một vector $v$:

### Ma trận $A$ (kích thước $n \times m$):
$$A = \begin{pmatrix} a_{11} & a_{12} & \cdots & a_{1m} \\ a_{21} & a_{22} & \cdots & a_{2m} \\ \vdots & \vdots & \ddots & \vdots \\ a_{n1} & a_{n2} & \cdots & a_{nm} \end{pmatrix}$$

### Vector $v$ (độ dài $m$):
$$v = \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_m \end{pmatrix}$$

Tích vô hướng $A \cdot v$ tạo ra một vector mới có độ dài $n$:
$$
A \cdot v = \begin{pmatrix} a_{11}v_1 + a_{12}v_2 + \cdots + a_{1m}v_m \\ a_{21}v_1 + a_{22}v_2 + \cdots + a_{2m}v_m \\ \vdots \\ a_{n1}v_1 + a_{n2}v_2 + \cdots + a_{nm}v_m \end{pmatrix}
$$

---

### Yêu cầu then chốt:

Số lượng **cột** của ma trận ($m$) phải bằng **độ dài** của vector ($m$). Nếu điều kiện này không được thỏa mãn, phép toán sẽ không xác định (undefined).

---

# Bài tập

Write a Python function that computes the dot product of a matrix and a vector. The function should return a list representing the resulting vector if the operation is valid, or -1 if the matrix and vector dimensions are incompatible.

You may assume the input matrix is a valid (non-jagged) list of lists and the vector is a non-empty list.

Example:

Input:

    a = [[1, 2], [2, 4]], b = [1, 2]

Output:

    [5, 10]

Reasoning:

    Row 1: (1 * 1) + (2 * 2) = 1 + 4 = 5; Row 2: (2 * 1) + (4 * 2) = 2 + 8 = 10

# Code

```python
def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	if len(a[0]) != len(b):
		return -1
	result = []
	for row in a:
		total = 0
		for i in range(len(row)):
			total += row[i] * b[i]
		result.append(total)
	return result
```

Dùng numpy

```python
import numpy as np

def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Check if dimensions are compatible
	if len(a[0]) != len(b):
		return -1
	
	# Convert to numpy arrays
	arr_a = np.array(a)
	arr_b = np.array(b)
	
	# Perform matrix-vector dot product
	result = arr_a @ arr_b
	
	# Convert result back to list
	return result.tolist()
```
