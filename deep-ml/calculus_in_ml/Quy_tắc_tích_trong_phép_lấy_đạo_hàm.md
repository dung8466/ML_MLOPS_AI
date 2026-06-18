Nhân hai đa thức $h(x)$ = $f(x)$ * $g(x)$

n = len(f), m = len(g) => Độ dài của đa thức tích sẽ là (n + m - 1)

Quy tắc số mũ: 

$$x^i \cdot x^j = x^{i+j}$$

Do đó, kết quả của `f[i] * g[j]` sẽ được cộng dồn vào vị trí `h[i + j]`.

Sau khi có đa thức tích $h(x)$, ta áp dụng quy tắc đạo hàm lũy thừa:

$$\frac{d}{dx}(cx^k) = k \cdot c \cdot x^{k-1}$$

```python
import numpy as np

def product_rule_derivative(f_coeffs: list, g_coeffs: list) -> list:   
    n = len(f_coeffs)
    m = len(g_coeffs)
    if n == 0 or m == 0:
        return []
    prod_len = n + m -1
    h_coeffs = [0.0] * prod_len
    for i in range(n):
        for j in range(m):
            h_coeffs[i + j] += f_coeffs[i] * g_coeffs[j]
    derivative_coeffs = []
    for k in range(1, prod_len):
        val = h_coeffs[k] * k
        derivative_coeffs.append(round(float(val), 4))
    if not derivative_coeffs:
        return [0.0]
    return derivative_coeffs
```
