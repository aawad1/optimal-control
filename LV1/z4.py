import numpy as np
import matplotlib.pyplot as plt

def LTI(E, F, C, D, x0, U):
    # Provjera dimenzija matrica i vektora
    n, m = F.shape
    r, _ = C.shape
    
    assert E.shape == (n, n), "Dimenzije matrice E nisu ispravne."
    assert D.shape == (r, m), "Dimenzije matrice D nisu ispravne."
    assert x0.shape == (n, 1), "Dimenzije vektora x0 nisu ispravne."
    assert U.shape[1] == m, "Dimenzije vektora U nisu ispravne."
    
    # Inicijalizacija spremnika za stanja i izlaze
    N = U.shape[0]
    X = np.zeros((N+1, n))
    Y = np.zeros((N, r))
    
    # Postavljanje pocetnog stanja
    X[0] = x0.flatten()
    
    # Izracunavanje trajektorije stanja i izlaza
    for k in range(N):
        X[k+1] = np.dot(E, X[k]) + np.dot(F, U[k])
        Y[k] = np.dot(C, X[k]) + np.dot(D, U[k])
    
    return X, Y

# Testiranje funkcije
E = np.array([[1.1269, -0.4940, 0.1129],
              [1, 0, 0],
              [0, 1, 0]])

F = np.array([[-0.3832, -0.3832],
              [0.5919, 0.8577],
              [0.5191, 0.4546]])

C = np.array([[1, 1, 0]])

D = np.array([[1, 1]])

x0 = np.array([[0],
               [0],
               [0]])

U = np.array([[1, 0],
              [0, 1]])

X, Y = LTI(E, F, C, D, x0, U)

# Iscrtavanje trajektorija stanja i izlaza
plt.plot(X[:, 0], label='x1')
plt.plot(X[:, 1], label='x2')
plt.plot(X[:, 2], label='x3')
plt.xlabel('Vrijeme')
plt.ylabel('Stanja')
plt.title('Trajektorije stanja')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(Y[:, 0], label='y')
plt.xlabel('Vrijeme')
plt.ylabel('Izlaz')
plt.title('Trajektorija izlaza')
plt.grid(True)
plt.legend()
plt.show()
