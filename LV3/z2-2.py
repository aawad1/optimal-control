import numpy as np
import matplotlib.pyplot as plt

def fib_search(f, c1, c2, N):
    # Definirajte Fibonaccijev niz
    fib = [1, 1]
    while fib[-1] < (c2 - c1) / 0.001:
        fib.append(fib[-1] + fib[-2])

    x_tr = []
    d_tr = []

    # Početni interval pretraživanja
    a, b = c1, c2

    for k in range(1, N + 1):
        # Trenutne tačke
        x1 = a + (fib[-k - 1] / fib[-k]) * (b - a)
        x2 = a + (fib[-k] / fib[-k]) * (b - a)

        if f(x1) < f(x2):
            b = x2
        else:
            a = x1

        # Dodajte trenutnu tačku pretraživanja i dužinu intervala
        x_tr.append((a + b) / 2)
        d_tr.append(b - a)

    x_min = (a + b) / 2
    f_min = f(x_min)

    return f_min, x_min, np.array(x_tr), np.array(d_tr)

def f1(x):
    return (x - 1)**2

def f2(x):
    return -x**4 + 10 * x**2

# Testiranje nad f1
c1, c2 = -5, 5
f_min, x_min, x_tr, d_tr = fib_search(f1, c1, c2, 10)

x_values = np.arange(c1, c2, 0.01)
f_values = f1(x_values)

plt.plot(x_values, f_values, 'k-', label='f(x)')
plt.plot(x_min, f_min, 'ro', label='Minimum', markersize=8)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Fibonaccijeva pretraga: f(x) = (x - 1)^2')
plt.legend()
plt.show()

# Testiranje nad f2
f_min, x_min, x_tr, d_tr = fib_search(f2, c1, c2, 10)

f_values = f2(x_values)

plt.plot(x_values, f_values, 'k-', label='f(x)')
plt.plot(x_min, f_min, 'ro', label='Minimum', markersize=8)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Fibonaccijeva pretraga: f(x) = -x^4 + 10*x^2')
plt.legend()
plt.show()
