import numpy as np

def _get_new_alpha(zeta, old_alpha):
    """

    :param zeta:
    :param old_alpha:
    :return:
    """
    new_alpha = old_alpha * zeta**2
    return new_alpha


def _get_new_normalization(alpha):
    """

    :param alpha:
    :return:
    """
    return np.power(2*alpha/np.pi, 3/4)


def use_sto_1g(zeta):
    """

    :param zeta:
    :param alpha:
    :return:
    """
    # alpha for 1g in zeta=1.0
    old_alpha = 0.270950
    alpha = _get_new_alpha(zeta, old_alpha)
    N = _get_new_normalization(alpha)
    return alpha, N

def overlap_int_gaussian(R_a, alpha_a, N1, R_b, alpha_b, N2):
    """

    :param R_a:
    :param alpha_a:
    :param R_b:
    :param alpha_b:
    :return:
    """
    p = alpha_a + alpha_b
    norm = np.linalg.norm(R_a - R_b)
    norm = norm**2
    K = np.exp(-alpha_a*alpha_b*norm / (alpha_a+alpha_b))
    S = np.power(np.pi/p, 3/2) * K * N1 * N2
    return S

def kinetic_int_gaussian(R_a, alpha_a, N1, R_b, alpha_b, N2):
    """

    :param R_a:
    :param alpha_a:
    :param N1:
    :param R_b:
    :param alpha_b:
    :param N2:
    :return:
    """
    S = overlap_int_gaussian(R_a, alpha_a, N1, R_b, alpha_b, N2)
    norm = np.linalg.norm(R_a - R_b)
    norm = norm**2
    c = alpha_a*alpha_b/(alpha_a + alpha_b)
    T = c*(3 - 2*c*norm)*S
    return T

def get_overlap_matrix(orbital_list):
    """

    :param orbital_list: dimension (N, 3),
    N the number of available orbitals, with corresponding center
    only support sto-1g now
    :return:
    """
    zeta = 1.24
    alpha, N_factor = use_sto_1g(zeta)
    N = len(orbital_list)
    S_matrix = np.zeros((N, N))
    for i_idx, R_i in enumerate(orbital_list):
        for j_idx, R_j in enumerate(orbital_list):
            S_matrix[i_idx][j_idx] = overlap_int_gaussian(R_i, alpha, N_factor,
                                                          R_j, alpha, N_factor)
    return S_matrix

def get_kinetic_matrix(orbital_list):
    zeta = 1.24
    alpha, N_factor = use_sto_1g(zeta)
    N = len(orbital_list)
    T_matrix = np.zeros((N, N))
    for i_idx, R_i in enumerate(orbital_list):
        for j_idx, R_j in enumerate(orbital_list):
            T_matrix[i_idx][j_idx] = kinetic_int_gaussian(R_i, alpha, N_factor,
                                                          R_j, alpha, N_factor)
    return T_matrix




def main():
    """
    find overlap integral
    :return:
    """
    zeta = 1.24
    alpha, N = use_sto_1g(zeta)
    R_list = np.array([
        [0.0, 0.0, 0.0],
        [1.4, 0.0, 0.0],
        [2.8, 0.0, 0.0]
    ])
    S_matrix = get_overlap_matrix(R_list)
    T_matrix = get_kinetic_matrix(R_list)
    print(S_matrix)
    print(T_matrix)



if __name__ == '__main__':
    main()
