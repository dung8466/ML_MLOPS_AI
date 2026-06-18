Dạng đa thức c * $x^{n}$ => đạo hàm: c * n * $x^{n-1}$

```python
def poly_term_derivative(c: float, x: float, n: float) -> float:
    # Your code here
    if n == 0:
        return 0.0
    if n == 1:
        return float(c)
    return float(c * x * n *(n-1))
```
