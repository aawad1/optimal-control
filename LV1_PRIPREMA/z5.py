import numpy as np
def is_prost(broj):
    if broj <= 2:
        return True
    for i in range(2, int(broj**0.5) + 1):
        if broj % i == 0:
            return False
    return True

def f_matrica(A):
    # Inicijalizacija matrice B iste dimenzionalnosti kao matrica A
    B = np.zeros_like(A, dtype=float)

    # Iteracija kroz matricu A i postavljanje odgovarajućih vrijednosti u matricu B
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] == 1:
                B[i, j] = 0.5  # Jedinica, postavi 0.5
            elif is_prost(A[i, j]):
                B[i, j] = 1  # Prost broj, postavi 1
            else:
                B[i, j] = 0  # Složen broj, postavi 1

    return B
