import math
def faktorijel(n):
    if n == 0 or n == 1:
        return 1
    else:
        rezultat = 1
        for i in range(2, n + 1):
            rezultat *= i
        return rezultat
        
def moj_sin(x, N):

    # Inicijalizacija rezultata
    rezultat = 0.0

    # Računanje vrijednosti sin(x) pomoću Taylorovog reda
    for n in range(N + 1):
        koeficijent = ((-1) ** n) / faktorijel(2 * n + 1)
        rezultat += koeficijent * (x ** (2 * n + 1))

    return rezultat
