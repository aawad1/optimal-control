import numpy as np
import matplotlib.pyplot as plt

# Definirajte funkciju
def y(x):
    return 1 - np.exp(np.sin(x) / x)

# Generirajte x vrijednosti od -6π do 6π s korakom π/4
x_values = np.arange(-6 * np.pi, 6 * np.pi, np.pi/4)

# Izračunajte y vrijednosti koristeći definiranu funkciju
y_values = y(x_values)

# Nacrtajte diskretne vrijednosti funkcije crvenim znakovima "×"
plt.scatter(x_values, y_values, marker='x', color='red')

# Dodajte mrežu (grid), oznake osa x i y, te naslov
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('$y(x) = 1 - e^{\\frac{\\sin x}{x}}$')

# Prikaz grafika
plt.show()
