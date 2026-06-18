# Lý thuyết

---

# Thay đổi hình dạng Ma trận (Reshaping a Matrix)

Thay đổi hình dạng ma trận (reshaping) bao gồm việc thay đổi kích thước (số hàng và số cột) của một ma trận mà không làm biến đổi dữ liệu gốc bên trong của nó. Đây là một thao tác thiết yếu trong nhiều tác vụ học máy (machine learning), nơi dữ liệu đầu vào cần được định dạng theo một cấu trúc cụ thể.

Ví dụ, hãy xét một ma trận $M$:

### Ma trận gốc $M$ (kích thước $2 \times 4$):

$$
M = 
\begin{pmatrix} 
1 & 2 & 3 & 4 \\ 
5 & 6 & 7 & 8 
\end{pmatrix}
$$

### Ma trận đã thay đổi hình dạng $M'$ với kích thước mới $(4, 2)$:

$$
M' = 
\begin{pmatrix} 
1 & 2 \\ 
3 & 4 \\ 
5 & 6 \\ 
7 & 8 
\end{pmatrix}
$$

---

### Lưu ý quan trọng:

Luôn đảm bảo rằng **tổng số lượng phần tử** không thay đổi trong suốt quá trình biến đổi hình dạng. 
*(Ví dụ ở trên: $2 \times 4 = 8$ phần tử và $4 \times 2 = 8$ phần tử. Nếu tổng số phần tử không khớp, phép toán sẽ không thực hiện được).*

---

# Bài tập

Write a Python function that reshapes a given matrix into a specified shape. if it cant be reshaped return back an empty list []
Example:
Input:

    a = [[1,2,3,4],[5,6,7,8]], new_shape = (4, 2)

Output:

    [[1, 2], [3, 4], [5, 6], [7, 8]]

Reasoning:

    The given matrix is reshaped from 2x4 to 4x2.

# Code

```python
import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	
	arr = np.array(a)
	if np.array(a).size != new_shape[0] * new_shape[1]:
    # Xử lý lỗi nếu cần
    	return []
	reshaped_matrix = arr.reshape(new_shape).tolist()
	return reshaped_matrix
  
```
