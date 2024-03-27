import math
import numpy as np
import matplotlib.pyplot as plt
from  mpl_toolkits import  mplot3d

def f(x1, x2):
    return (x1**2) + 2*(x2**2)

x1=np.linspace(-2, 2, 50)
x2=np.linspace(-2, 2, 50)

X1, X2 = np.meshgrid(x1, x2)
Z = f(X1,X2)

# (a) 
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X1, X2, Z, 50)
ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_zlabel('$Z$')

# (b)
x1_region = np.linspace(-2, 2, 100)
x2_region = np.linspace(-2, 2, 100)
X1_region, X2_region = np.meshgrid(x1_region, x2_region)
region_mask = (X1_region**2 + X2_region**2 <= 2) & (X1_region - X2_region <= 1) & (X1_region >= 0)

# (b) Contour plot funkcije f preklopljen sa region plotom sa ograničenjima
plt.figure()
# Contour plot funkcije f
plt.contour(X1, X2, Z, levels=50)
# Region plot ograničenja
plt.contourf(region_mask, colors=['green'], alpha=0.5)
#plt.plot(x1_region, x1_region - 1, 'r--')
plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.title('Contour plot funkcije $f(x_1, x_2)$ sa region plotom ograničenja')
plt.grid(True)

# (c) Contour plotovi prvih parcijalnih izvoda funkcije f po svim njenim varijablama uz ograničenja
plt.figure()
# Prvi parcijalni izvod po x1
Z_partial_x1 = 2 * X1
plt.contour(X1, X2, Z_partial_x1, levels=50, colors='r')
#plt.contourf(region_mask, colors=['green'], alpha=0.5)

plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.title('Contour plot parcijalnog izvoda po $X_1$ sa ograničenjima')
plt.grid(True)
plt.show()

plt.figure()
# Prvi parcijalni izvod po x2
Z_partial_x2 = 4*X2**2
plt.contour(X1, X2, Z_partial_x2, levels=50, colors='g')
#plt.contourf(region_mask, colors=['green'], alpha=0.5)

plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.title('Contour plot parcijalnog izvoda po $X_2$ sa ograničenjima')
plt.grid(True)
plt.show()
