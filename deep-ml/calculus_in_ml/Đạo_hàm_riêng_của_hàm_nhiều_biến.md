# Công thức Đạo hàm (Toán học)

Dưới đây là bảng phân tích đạo hàm riêng cho từng hàm để bạn nắm vững bản chất:

1.  **`poly2d`**: $f(x,y) = x^2y + xy^2$
    *   $\frac{\partial f}{\partial x} = 2xy + y^2$
    *   $\frac{\partial f}{\partial y} = x^2 + 2xy$

2.  **`exp_sum`**: $f(x,y) = e^{x+y}$
    *   Sử dụng quy tắc chuỗi: $(e^u)' = e^u \cdot u'$
    *   $\frac{\partial f}{\partial x} = e^{x+y} \cdot \frac{\partial (x+y)}{\partial x} = e^{x+y}$
    *   $\frac{\partial f}{\partial y} = e^{x+y}$

3.  **`product_sin`**: $f(x,y) = x \sin(y)$
    *   $\frac{\partial f}{\partial x} = \sin(y)$ (Coi $\sin(y)$ là hằng số)
    *   $\frac{\partial f}{\partial y} = x \cos(y)$ (Coi $x$ là hằng số)

4.  **`poly3d`**: $f(x,y,z) = x^2y + yz^2$
    *   $\frac{\partial f}{\partial x} = 2xy$
    *   $\frac{\partial f}{\partial y} = x^2 + z^2$
    *   $\frac{\partial f}{\partial z} = 2yz$

5.  **`squared_error`**: $f(x,y) = (x-y)^2$
    *   $\frac{\partial f}{\partial x} = 2(x-y) \cdot 1 = 2(x-y)$
    *   $\frac{\partial f}{\partial y} = 2(x-y) \cdot (-1) = -2(x-y)$


---

#Code

```python
import numpy as np

def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> tuple[float, ...]:
    """
    Tính đạo hàm riêng của các hàm nhiều biến tại một điểm cụ thể.
    """
    
    if func_name == 'poly2d':
        # f(x,y) = x²y + xy²
        x, y = point
        df_dx = 2*x*y + y**2
        df_dy = x**2 + 2*x*y
        return (float(df_dx), float(df_dy))

    elif func_name == 'exp_sum':
        # f(x,y) = e^(x+y)
        x, y = point
        val = np.exp(x + y)
        df_dx = val
        df_dy = val
        return (float(df_dx), float(df_dy))

    elif func_name == 'product_sin':
        # f(x,y) = x·sin(y)
        x, y = point
        df_dx = np.sin(y)
        df_dy = x * np.cos(y)
        return (float(df_dx), float(df_dy))

    elif func_name == 'poly3d':
        # f(x,y,z) = x²y + yz²
        x, y, z = point
        df_dx = 2*x*y
        df_dy = x**2 + z**2
        df_dz = 2*y*z
        return (float(df_dx), float(df_dy), float(df_dz))

    elif func_name == 'squared_error':
        # f(x,y) = (x-y)²
        x, y = point
        df_dx = 2 * (x - y)
        df_dy = -2 * (x - y)
        return (float(df_dx), float(df_dy))

    else:
        raise ValueError("func_name không hợp lệ.")

# Ví dụ chạy thử:
# Với poly2d tại điểm (1.0, 2.0):
# df/dx = 2(1)(2) + 2^2 = 4 + 4 = 8.0
# df/dy = 1^2 + 2(1)(2) = 1 + 4 = 5.0
print(compute_partial_derivatives('poly2d', (1.0, 2.0))) # Kết quả: (8.0, 5.0)
```

---

