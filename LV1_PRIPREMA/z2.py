import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definirajte funkciju
def z(x, y):
    r = np.sqrt(x**2 + y**2)
    if (r.any() != 0):
        return np.sin(r) / r  
    else: 
        return 0

# Generirajte x i y vrijednosti s koracima ∆x = ∆y = 0.1
x_values = np.arange(-10, 10, 0.1)
y_values = np.arange(-10, 10, 0.1)

# Kreirajte mrežu x i y vrijednosti
x, y = np.meshgrid(x_values, y_values)

# Izračunajte z vrijednosti koristeći definiranu funkciju
z_values = z(x, y)

# Prikaz funkcije pomoću surface plot
fig = plt.figure(figsize=(12, 6))

# Prikaz surface plot
ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_surface(x, y, z_values, cmap='viridis', rstride=5, cstride=5, alpha=0.8)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('Surface Prikaz funkcije z(x, y)')
ax1.set_zlim(-0.5, 1)
ax1.grid(True)

# Dodajte legendu
fig.colorbar(surf, ax=ax1, orientation='vertical', label='z')

# Prikaz funkcije pomoću contour plot
ax2 = fig.add_subplot(122)
contour = ax2.contourf(x, y, z_values, cmap='viridis')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Contour Prikaz funkcije z(x, y)')
ax2.grid(True)

# Dodajte legendu
fig.colorbar(contour, ax=ax2, orientation='vertical', label='z')

# Prikaz grafova
plt.tight_layout()
plt.show()
