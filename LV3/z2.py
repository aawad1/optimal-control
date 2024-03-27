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

def fib_search(f, c1, c2, N):
    F=(c2-c1)/0.001
    n=fibonacci(F)
    print(n)
    fib = [1, 1]
    for k in range(3,n+1):
        fib.append(fib[-1]+fib[-2])
    x_tr = []
    d_tr = []
    a, b = c1, c2
    for k in range(1,N):
        c = a+(b-a)*(1-fib[n-1-k]/fib[n-k])
        d = a+(b-a)*(fib[n-1-k]/fib[n-k])
        x_tr.append((a+b)/2)
        d_tr.append(b-a)
        if(f(c)<=f(d)):
            b = d
        else:
            a = c
    x_min=(a+b)/2
    f_min=f(x_min)
    return f_min, x_min, x_tr, d_tr
    
# f1

def f1(x):
    return (x - 1)**2

c1, c2 = -5, 5
f_min, x_min, x_tr, d_tr = fib_search(f1, c1, c2, 20)

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
print(x_tr)
print(d_tr)

# f2
def f2(x):
    return -x**4 + 10 * x**2

f_min, x_min, x_tr, d_tr = fib_search(f2, c1, c2, 20)

f_values = f2(x_values)

plt.plot(x_values, f_values, 'k-', label='f(x)')
plt.plot(x_min, f_min, 'ro', label='Minimum', markersize=8)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Fibonaccijeva pretraga: f(x) = -x^4 + 10*x^2')
plt.legend()
plt.show()
