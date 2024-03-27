import numpy as np
import matplotlib.pyplot as plt

def uni_search(f, c1, c2, N):

    F = (c2 - c1) / 0.01
    fib = [1, 1]
    while fib[-1] < F:
        fib.append(fib[-1] + fib[-2])
    n =  len(fib)
    
    y_tr = []
    yn = np.arange(c1, c2, 0.01)
    for j in yn:
        x_tr = []
        x_min = []
        f_min = []
        a, b = c1, c2
        for k in range(1, N):
            x1 = a + (b-a)*(1-fib[n-1-k]/fib[n-k])
            x2 = a + (b-a)*(fib[n-1-k]/fib[n-k])
            
            if f(x1, j) < f(x2, j):
                b = x2
            else:
                a = x1

            x_tr.append((a + b) / 2)
        x_min.append((a + b) / 2)
        f_min.append(f(x_min[-1], j))
        
        y_tr.append(x_tr)
    
    return np.array(f_min), np.array(x_min), np.array(y_tr)


def f1(x, y):
    return (x - 2)**2 + 2 * (y - 3)**2

def f2(x, y):
    return 100 * (x**2 - y)**2 + (1 - x)**2

# Testiranje nad f1
c1, c2 = -5, 5
f_min, x_min, x_tr = uni_search(f1, c1, c2, 10)

x_values = np.arange(c1, c2, 0.01)
y_values = np.arange(c1, c2, 0.01)
X, Y = np.meshgrid(x_values, y_values)
Z = f1(X, Y)

plt.contour(X, Y, Z, colors='black')
plt.plot(x_min[0], x_min[1], 'ro')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Univarijantno pretraživanje: f(X, Y) = (x - 2)^2 + 2 * (y - 3)^2')
plt.grid(True)
plt.show()

# Testiranje nad f2
f_min, x_min, x_tr = uni_search(f2, c1, c2, 10)

Z = f2(X, Y)

plt.contour(X, Y, Z, colors='black')
plt.plot(x_min[0], x_min[1], 'ro')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Univarijantno pretraživanje: f(X, Y) = 100 * (x^2 - y)^2 + (1 - x)^2')
plt.grid(True)
plt.show()
