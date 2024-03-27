import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**3 - np.cos(5*x)

l1 = np.linspace(-1, -0.2, 20)
l2 = np.linspace(-0.2, 0.4, 20)
l3 = np.linspace(0.4, 1, 20)

y1=f(l1)
y2=f(l2)
y3=f(l3)

plt.plot(l1, y1, color='blue')
plt.plot(l2, y2, color='green')
plt.plot(l3, y3, color='black')

segments = [(l1, y1), (l2, y2), (l3, y3)]
markers = ['x', 'o']

for segment in segments:
    x_values, y_values = segment
    min_index = np.argmin(y_values)
    max_index = np.argmax(y_values)
    min_x, min_y = x_values[min_index], y_values[min_index]
    max_x, max_y = x_values[max_index], y_values[max_index]
    
    plt.plot(min_x, min_y, color='blue', marker='x', markersize=8)
    plt.plot(max_x, max_y, color='red', marker='o', markersize=8)

plt.xlabel('$t$')
plt.ylabel('y(t)')
plt.title('Funkcija')
plt.grid(True)
plt.show()