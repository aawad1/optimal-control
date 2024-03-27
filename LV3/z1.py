import numpy as np
import matplotlib.pyplot as plt

def NR(f, df, ddf, x0, N):
    x_tr = [x0]
    for i in range(N):
        x = x_tr[-1]
        x_new = x - f(x) / df(x)
        x_tr.append(x_new)
        if abs(x_new - x) < 1e-6:
            break
    return f(x_new), x_new, np.array(x_tr)

def f1(x):
    return (x - 1)**2

def df1(x):
    return 2 * (x - 1)

def ddf1(x):
    return 2

def f2(x):
    return -x**4 + 10 * x**2

def df2(x):
    return -4 * x**3 + 20 * x

def ddf2(x):
    return -12 * x**2 + 20

# f1
x_values = np.arange(-5, 5, 0.01)
f_values = f1(x_values)
f_min, x_min, x_tr = NR(f1, df1, ddf1, 0, 100)
plt.plot(x_values, f_values, 'k-', label='f(x)')
plt.plot(x_min, f_min, 'ro', label='Minimum', markersize=8)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Newton-Raphson: f(x) = (x - 1)^2')
plt.legend()
plt.show()

# f2
x_values = np.arange(-5, 5, 0.01)
f_values = f2(x_values)
f_min, x_min, x_tr = NR(f2, df2, ddf2, 0.5, 100)
plt.plot(x_values, f_values, 'k-', label='f(x)')
plt.plot(x_min, f_min, 'ro', label='Minimum', markersize=8)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Newton-Raphson: f(x) = -x^4 + 10*x^2')
plt.legend()
plt.show()
