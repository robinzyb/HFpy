from element import H
from integration import get_fock_prime_matrix, get_kinetic_matrix, \
    get_nuclear_matrix, get_density_matrix, get_nn_energy, get_elec_energy
import numpy as np
import matplotlib.pyplot as plt


def run_scf(system, N_e, C_0, threshold):
    iter = 0
    diff = 100
    while diff > threshold:
        F_prime, F, X = get_fock_prime_matrix(system, N_e, C_0)
        _, C_prime = np.linalg.eig(F_prime)
        H = get_kinetic_matrix(system) + get_nuclear_matrix(system)
        C_1 = np.matmul(X, C_prime)
        P_0 = get_density_matrix(C_0, N_e)
        P_1 = get_density_matrix(C_1, N_e)
        diff = np.abs(C_1 - C_0).max()
        C_0 = C_1
        E = get_elec_energy(P_1, H, F) + get_nn_energy(system)
        iter += 1
        print("now is iteration {0}, the convegence is {1}".format(iter, diff))
        print("Total Energy: {0}".format(E))
    print(C_1)
    return E


def main():
    """
    find overlap integral
    :return:
    """
    C = np.array(
        [
            [1, 0],
            [0, 1.0]
        ])
    system = [
        H("sto-3g", [0.0, 0.0, 0.0]),
        H("sto-3g", [2.1, 0.0, 0.0])
    ]
    # the total electron number
    N_e = 2
    e = run_scf(system, N_e, C, 1e-6)




if __name__ == '__main__':
    main()
