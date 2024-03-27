import math
import numpy as np
import matplotlib.pyplot as plt
from  mpl_toolkits import  mplot3d

def f(x1, x2):
    return 4*(x1**2) + (x2**2) + 16*(x1**2)*(x2**2)

x1=np.linspace(-0.5, 0.5, 20)
x2=np.linspace(-1, 1, 20)

X, Y = np.meshgrid(x1, x2)
Z = f(X,Y)

# (a) 
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50)
ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_zlabel('$Z$')

# (b) 
plt.figure()
plt.contour(X, Y, Z, levels=50)
plt.xlabel('$X$')
plt.ylabel('$Y$')
plt.title('Contour plot funkcije $f$ sa zadatim ograničenjima po varijablama')
plt.grid(True)

# (c) 
plt.figure()
# Parcijalni izvodi po x1
plt.contour(X, Y, 8*X + 32*X*Y**2, levels=50, colors='r', linestyles='dashed')
plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.title('Contour plot parcijalnog izvoda po $X$')
plt.grid(True)

plt.figure()
# Parcijalni izvodi po x2
plt.contour(X, Y, 2*Y + 32*X**2*Y, levels=50, colors='g', linestyles='dashed')
plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.title('Contour plot parcijalnog izvoda po $Y$')
plt.grid(True)

plt.show()