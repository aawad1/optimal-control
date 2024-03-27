import numpy as np
import matplotlib.pyplot as plt

def y(x):
    return 1 - np.exp(np.sin(x) / x)

x_values = np.linspace(-6 * np.pi, 6 * np.pi, 50)

y_values = y(x_values)

# Nacrtajte grafik funkcije s zelenom isprekidanom linijom
plt.plot(x_values, y_values, 'g--')

# Dodajte mrežu (grid), oznake osa x i y, te naslov
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('$y(x) = 1 - e^{\\frac{\\sin x}{x}}$')

# Prikaz grafika
plt.show()
