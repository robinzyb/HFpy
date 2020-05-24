"""
basis_name = {
   element: [
    basis_1 = [[contract_factor1, alpha1]],
    basis_2 = [[contract_factor1, alpha1]]
   ]
}
"""
import numpy as np
from utils import get_normalization

sto_1g = {
    "H": np.array([
        [[1.0, 0.270950]]
    ])
}

#zeta = 1.24
sto_3g = {
    "H": np.array([
        [[0.444635, 0.168856], [0.535328, 0.623913], [0.154329, 3.42525]]
    ])
}
class gto():
    def __init__(self, d, alpha, position):
        setattr(self, "d", d)
        setattr(self, "alpha", alpha)
        setattr(self, "N", get_normalization(alpha))
        setattr(self, "position", position)


