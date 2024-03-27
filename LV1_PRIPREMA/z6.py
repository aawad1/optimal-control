import numpy as np
import matplotlib.pyplot as plt

def funkcija(funkcija, opseg, k, x0):
    # Generiranje vektora x na opsegu opseg s korakom k, počevši od x0
    x_values = np.arange(x0, opseg[1] + k, k)

    # Izračunavanje vektora y pomoću zadane funkcije
    y_values = funkcija(x_values)

    # Indeksi koji odgovaraju opsegu
    indeksi_opsega = np.where((x_values >= opseg[0]) & (x_values <= opseg[1]))[0]

    # Prikaz funkcije na grafiku
    plt.plot(x_values[indeksi_opsega], y_values[indeksi_opsega], label='f(x)')

    # Označavanje minimalne i maksimalne vrijednosti zelenim znakom
    min_index = np.argmin(y_values[indeksi_opsega])
    max_index = np.argmax(y_values[indeksi_opsega])

    plt.scatter(x_values[indeksi_opsega][min_index], y_values[indeksi_opsega][min_index], c='green', marker='o', label='Min')
    plt.scatter(x_values[indeksi_opsega][max_index], y_values[indeksi_opsega][max_index], c='green', marker='o', label='Max')

    # Dodavanje mreže (grid), oznaka osa x i y, te naslov
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Grafik funkcije')

    # Legenda
    plt.legend()

    # Prikaz grafika
    plt.show()

