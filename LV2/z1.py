from sympy import symbols, solve
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


x, y = symbols('x y')

f = x**2 + 3*y**2

constraint1 = x**2 + y**2 - 3
constraint2 = 1 - x - y 
constraint3 = -1*x

solutions = solve([f.diff(x), f.diff(y), constraint1, constraint2, constraint3], (x, y, 'lambda1', 'lambda2', 'lambda3'), dict=True)

for solution in solutions:
    x_val = solution[x]
    y_val = solution[y]
    print("Stacionarna tačka (x, y):", (x_val, y_val))
    print("Vrednost funkcije u stacionarnoj tački f(x, y):", f.subs({x: x_val, y: y_val}))
    print("Vrednosti Lagrangeovih multiplikatora:", solution['lambda1'], solution['lambda2'], solution['lambda3'])

# Odabir samo realnih rešenja sa nenegativnim slack varijablama
real_solutions = [solution for solution in solutions if solution[x].is_real and solution[y].is_real and solution['lambda1'] >= 0 and solution['lambda2'] >= 0 and solution['lambda3'] >= 0]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_vals = np.linspace(-3, 3, 100)
y_vals = np.linspace(-3, 3, 100)
x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)
z_mesh = x_mesh**2 + 3*y_mesh**2

ax.plot_surface(x_mesh, y_mesh, z_mesh, cmap='viridis')

ax.set_xlabel('X osa')
ax.set_ylabel('Y osa')
ax.set_zlabel('Z osa')

plt.show()

fig2 = plt.figure()
x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 3, 400)
x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)
z_mesh = x_mesh**2 + 3*y_mesh**2

plt.contour(x_mesh, y_mesh, z_mesh, levels=50, cmap='viridis')
plt.xlim([-3, 3])
plt.ylim([-3, 3])

# Prikaz dozvoljenih vrednosti
plt.fill_between(x_vals, 1-x_vals, 3-x_vals**2, where=(x_vals>=0) & (x_vals<=1), color='grey', alpha=0.5)

# Prikaz stacionarnih tačaka
for solution in real_solutions:
    plt.plot(solution[x], solution[y], 'o', color='blue')

plt.xlabel('X osa')
plt.ylabel('Y osa')
plt.grid(True)
plt.show()
