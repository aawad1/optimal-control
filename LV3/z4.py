import numpy as np
import matplotlib.pyplot as plt

def fibonacci(F):
    f1=1
    f2=1
    for i in range(int(F-1)):
        f=f1+f2
        if(F>f2 and F<f):
            return i+2
        f1=f2
        f2=f

def uni_search(f, c1, c2, N):

    F = (c2 - c1) / 0.01
    #n = fibonacci(F)
    fib = [1, 1]
    while fib[-1] < F:
        fib.append(fib[-1] + fib[-2])

    x_tr = []
    a, b = c1, c2
    c, d = c1, c2
    
    for k in range(1, N):
        # Trenutne tačke
        x1 = a + (b-a)*(1-fib[n-1-k]/fib[n-k])
        x2 = a + (b-a)*(fib[n-1-k]/fib[n-k])
        y1 = c + (d-c)*(1-fib[n-1-k]/fib[n-k])
        y2 = c + (d-c)*(fib[n-1-k]/fib[n-k])
        
        if f(x1, y1) < f(x2, y2):
            b, d = x2, y2
        else:
            a, c = x1, y1

        # Dodajte trenutnu tačku pretraživanja
        x_tr.append(((a + b) / 2, (c + d)/2))

    x_min = (a + b) / 2
    f_min = f(*x_min)

    return f_min, x_min, np.array(x_tr)

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
