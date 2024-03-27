import numpy as np
def get_parni_neparni(v):
    if not isinstance(v, (list, np.ndarray)):
        raise ValueError("Argument v mora biti lista ili numpy array.")

    broj_parnih = 0
    broj_neparnih = 0

    for elem in v:
        if elem % 2 == 0:
            broj_parnih += 1
        else:
            broj_neparnih += 1

    return broj_parnih, broj_neparnih
