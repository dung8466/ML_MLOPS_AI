Để tính đạo hàm của hàm phân thức $f(x) = \frac{g(x)}{h(x)}$ tại một điểm $x$ cho trước, chúng ta sẽ áp dụng **Quy tắc đạo hàm của một thương (Quotient Rule)**.

---

### Hiểu Toán trước: Quy tắc thương

Công thức đạo hàm của một thương là:
$$f'(x) = \left( \frac{g(x)}{h(x)} \right)' = \frac{g'(x)h(x) - g(x)h'(x)}{[h(x)]^2}$$

**Các bước logic để lập trình:**
1.  **Tính $g(x)$ và $h(x)$**: Thay giá trị $x$ vào đa thức gốc.
2.  **Tìm $g'(x)$ và $h'(x)$**: Tính đạo hàm của từng đa thức, sau đó thay giá trị $x$ vào.
3.  **Áp dụng công thức**: Ráp các giá trị vừa tính được vào công thức quy tắc thương.
4.  **Lưu ý**: Nếu $h(x) = 0$, đạo hàm không xác định (lỗi chia cho 0).

---

```python
def evaluate_polynomial(coeffs: list, x: float) -> float:
    """Hàm phụ để tính giá trị của đa thức tại điểm x."""
    res = 0
    n = len(coeffs)
    for i, c in enumerate(coeffs):
        power = n - 1 - i
        res += c * (x ** power)
    return res

def derivative_coefficients(coeffs: list) -> list:
    """Hàm phụ để tìm các hệ số của đa thức đạo hàm."""
    n = len(coeffs)
    deriv_coeffs = []
    for i, c in enumerate(coeffs):
        power = n - 1 - i
        if power > 0:
            deriv_coeffs.append(c * power)
    # Nếu đa thức gốc là hằng số, đạo hàm là [0]
    return deriv_coeffs if deriv_coeffs else [0]

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    # 1. Tính giá trị g(x) và h(x)
    gx = evaluate_polynomial(g_coeffs, x)
    hx = evaluate_polynomial(h_coeffs, x)
    
    # Kiểm tra mẫu số
    if hx == 0:
        raise ValueError("Mẫu số h(x) bằng 0 tại điểm x, đạo hàm không xác định.")
    
    # 2. Tính giá trị g'(x) và h'(x)
    g_prime_coeffs = derivative_coefficients(g_coeffs)
    h_prime_coeffs = derivative_coefficients(h_coeffs)
    
    g_prime_x = evaluate_polynomial(g_prime_coeffs, x)
    h_prime_x = evaluate_polynomial(h_prime_coeffs, x)
    
    # 3. Áp dụng công thức quy tắc thương
    # f'(x) = (g'h - gh') / h^2
    f_prime_x = (g_prime_x * hx - gx * h_prime_x) / (hx ** 2)
    
    return float(f_prime_x)
```

---

