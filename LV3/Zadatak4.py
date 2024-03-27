from sympy import symbols, solve, lambdify
import numpy as np
import matplotlib.pyplot as plt

# Deklaracija simbola
x, y, l1, l2, l3 = symbols('x y l1 l2 l3')

# Definicija funkcije kriterija
f = x**2 + 3*y**2

# Definicija ograničenja
constraint1 = x**2 + y**2 - 3
constraint2 = x + y - 1
constraint3 = x

# Pretvaranje simbola u funkcije
constraint1_func = lambdify((x, y), constraint1, 'numpy')
constraint2_func = lambdify((x, y), constraint2, 'numpy')
constraint3_func = lambdify((x, y), constraint3, 'numpy')

# Formulacija Lagrangeove funkcije
L = f - l1*constraint1 - l2*constraint2 - l3*constraint3

# Rješavanje sustava jednadžbi
solutions = solve([L.diff(var) for var in (x, y, l1, l2, l3)], (x, y, l1, l2, l3), dict=True)

# Filtriranje realnih rješenja
real_solutions = []
for sol in solutions:
    if all(sol[var].is_real for var in sol):
        real_solutions.append(sol)

# Filtriranje rješenja s nenegativnim slack varijablama
valid_solutions = []
for sol in real_solutions:
    if all(sol[var] >= 0 for var in sol if var != l1 and var != l2 and var != l3):
        valid_solutions.append(sol)

# Ispis stacionarnih tačaka
for i, sol in enumerate(valid_solutions):
    print(f"Stacionarna tačka {i+1}: x = {sol[x]}, y = {sol[y]}")

# Prikaz površine funkcije kriterija u 3D prostoru
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_vals = np.linspace(-3, 3, 100)
y_vals = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = X**2 + 3*Y**2

ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_xlabel('X osa')
ax.set_ylabel('Y osa')
ax.set_zlabel('Z osa')

plt.show()

# Prikaz konturnog plot-a
x_vals = np.linspace(-3, 3, 100)
y_vals = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_vals, y_vals)
valid_region = (constraint1_func(X, Y) <= 0) & (constraint2_func(X, Y) <= 0) & (constraint3_func(X, Y) <= 0)

Z = X**2 + 3*Y**2

plt.contour(X, Y, Z, 100, cmap='RdGy')
plt.xlabel('X osa')
plt.ylabel('Y osa')

# Prikaz skupa dozvoljenih vrijednosti
plt.imshow(valid_region.astype(int), 
           extent=(x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()), origin="lower", cmap="Greys", alpha=0.7)

# Prikaz tačaka minimuma i maksimuma
for sol in valid_solutions:
    plt.plot(sol[x], sol[y], 'o', color='blue')

plt.grid(True)
plt.show()
