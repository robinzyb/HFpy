"""
collection of gaussian integration
"""
import numpy
import numpy as np
from utils import get_basis_list, F_0, get_orth_basis


def s_int_gto(gto1, gto2):
    """
    to get the overlap integral between two gto
    :param gto1: gto object
    :param gto2: gto object
    :return:
    """
    p = gto1.alpha + gto2.alpha
    l2norm = np.linalg.norm(gto1.position - gto2.position)**2
    K = np.exp(-gto1.alpha * gto2.alpha * l2norm / (gto1.alpha+gto2.alpha))
    S = np.power(np.pi/p, 3/2) * K * gto1.N * gto2.N
    return S


def s_int_basis(basis1, basis2):
    """
    to get the overlap integral between two basis
    basis is a list of gto
    :param basis1:
    :param basis2:
    :return:
    """
    S=0
    for gto1 in basis1:
        for gto2 in basis2:
            S += gto1.d * gto2.d * s_int_gto(gto1, gto2)
    return S


def get_overlap_matrix(system):
    """
    get the whole overlap matrix
    :param system: a list of element object
    :return:
    """
    #expand the element object into a basis_list
    basis_list = get_basis_list(system)
    N = len(basis_list)
    S_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            S_matrix[i][j] = s_int_basis(basis_list[i], basis_list[j])
    return S_matrix


def t_int_gto(gto1, gto2):
    """
    get the kinetic integration with two gaussian function
    :param gto1:
    :param gto2:
    :return:
    """
    s = s_int_gto(gto1, gto2)
    l2norm = np.linalg.norm(gto1.position - gto2.position)**2
    c = gto1.alpha*gto2.alpha/(gto1.alpha + gto2.alpha)
    t = c*(3 - 2*c*l2norm)*s
    return t

def t_int_basis(basis1, basis2):
    """
    to get the kinetic integral between two basis
    basis is a list of gto
    :param basis1:
    :param basis2:
    :return:
    """
    T=0
    for gto1 in basis1:
        for gto2 in basis2:
            T += gto1.d * gto2.d * t_int_gto(gto1, gto2)
    return T

def get_kinetic_matrix(system):
    """
    get the whole kinetic matrix
    :param system: a list of element object
    :return:
    """
    #expand the element object into a basis_list
    basis_list = get_basis_list(system)
    N = len(basis_list)
    T_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            T_matrix[i][j] = t_int_basis(basis_list[i], basis_list[j])
    return T_matrix


def nuc_int_gto(gto1, gto2, element):
    """
    find the coulomb integral between 2 gtos and nuclear charge
    :param gto1:
    :param gto2:
    :param element: element object
    :return:
    """
    p = gto1.alpha + gto2.alpha
    l2norm = np.linalg.norm(gto1.position - gto2.position)**2
    c = gto1.alpha*gto2.alpha/p
    K = np.exp(-c * l2norm)
    R_p = (gto1.alpha*gto1.position + gto2.alpha*gto2.position)/p
    # get the variable for F0 function
    t = p * np.linalg.norm(R_p - element.position)**2
    v_nuc = -2 * np.pi * element.charge * K * F_0(t) * gto1.N * gto2.N / p
    return v_nuc

def nuc_int_basis(basis1, basis2, element):
    """
    to get the nuclear integral between two basis
    basis is a list of gto
    :param basis1:
    :param basis2:
    :param element:
    :return:
    """
    V_nuc=0
    for gto1 in basis1:
        for gto2 in basis2:
            V_nuc += gto1.d * gto2.d * nuc_int_gto(gto1, gto2, element)
    return V_nuc

def get_nuclear_matrix(system):
    """
    get the whole nuclear matrix
    :param system: a list of element object
    :return:
    """
    #expand the element object into a basis_list
    basis_list = get_basis_list(system)
    N = len(basis_list)
    V_nuc_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            for element in system:
                V_nuc_matrix[i][j] += nuc_int_basis(basis_list[i], basis_list[j], element)
    return V_nuc_matrix

def get_h_matrix(system):
    """

    :param system: a list of elements
    :return:

    """
    return get_kinetic_matrix(system) + get_nuclear_matrix(system)


def ee_int_gto(gto1, gto2, gto3, gto4):
    """
    find the four center integral
    :param gto1:
    :param gto2:
    :param gto3:
    :param gto4:
    :return:
    """
    k = ((gto1.alpha + gto2.alpha)*
         (gto3.alpha + gto4.alpha)*
         (gto1.alpha + gto2.alpha + gto3.alpha + gto4.alpha)**0.5)
    l2norm1 = np.linalg.norm(gto1.position - gto2.position)**2
    l2norm2 = np.linalg.norm(gto3.position - gto4.position)**2
    alpha_1 = gto1.alpha * gto2.alpha/(gto1.alpha + gto2.alpha)
    alpha_2 = gto3.alpha * gto4.alpha/(gto3.alpha + gto4.alpha)
    K1 = np.exp(-alpha_1 * l2norm1)
    K2 = np.exp(-alpha_2 * l2norm2)
    R_p = ((gto1.alpha * gto1.position + gto2.alpha * gto2.position)/
           (gto1.alpha + gto2.alpha))
    R_q = ((gto3.alpha * gto3.position + gto4.alpha * gto4.position)/
           (gto3.alpha + gto4.alpha))
    norm3 = np.linalg.norm(R_p - R_q)**2
    t = ((gto1.alpha + gto2.alpha)*
          (gto3.alpha + gto4.alpha)/
          (gto1.alpha + gto2.alpha + gto3.alpha + gto4.alpha) * norm3)
    v_ee = 2 * np.pi**2.5 * K1 * K2 * F_0(t) * gto1.N * gto2.N * gto3.N * gto4.N / k
    return v_ee

def ee_int_basis(basis1, basis2, basis3, basis4):
    """
    
    :param basis1: 
    :param basis2: 
    :param basis3: 
    :param basis4: 
    :return: 
    """
    V_ee=0
    for gto1 in basis1:
        for gto2 in basis2:
            for gto3 in basis3:
                for gto4 in basis4:
                    V_ee += (gto1.d * gto2.d *
                             gto3.d * gto4.d *
                             ee_int_gto(gto1, gto2, gto3, gto4))
    return V_ee

def get_ee_matrix(system):
    """

    :param system: a list of elements
    :return:
    """
    #expand the element object into a basis_list
    basis_list = get_basis_list(system)
    N = len(basis_list)
    V_ee_matrix = np.zeros((N, N, N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    V_ee_matrix[i][j][k][l] = ee_int_basis(basis_list[i],
                                                           basis_list[j],
                                                           basis_list[k],
                                                           basis_list[l])
    return V_ee_matrix


def get_density_matrix(C, N):
    """

    :param C: coeffecient matrix
    :return:

    """
    N_occ = int(N/2)
    if N_occ == 1:
        C = C[:, [0]]
    else:
        C = C[:, :N_occ]

    P = np.matmul(C, C.T)
    P = P*2.0
    return P

def get_G_matrix(system, N_e, C):
    V_ee = get_ee_matrix(system)
    P = get_density_matrix(C, N_e)
    basis_list = get_basis_list(system)
    N = len(basis_list)
    G = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    G[i][j] += P[k][l] * (V_ee[i][j][l][k] - 0.5 * V_ee[i][k][l][j])
    return G


def get_fock_matrix(system, N_e, C):
    G = get_G_matrix(system, N_e, C)
    H = get_kinetic_matrix(system) + get_nuclear_matrix(system)
    F_matrix = H + G
    return F_matrix


def get_fock_prime_matrix(system, N_e, C):
    F = get_fock_matrix(system, N_e, C)
    S = get_overlap_matrix(system)
    X = get_orth_basis(S)
    F_prime = np.matmul(X.T, np.matmul(F, X))
    return F_prime, F, X

def get_nn_energy(system):
    V_nn = 0
    for ele1 in system[:-1]:
        for ele2 in system[1:]:
            R_12 = np.linalg.norm(ele1.position - ele2.position)
            V_nn += ele1.charge * ele2.charge / R_12
    return V_nn


def get_elec_energy(P, H, F):
    E = 0.0
    N = len(P)
    #electronic energy
    for i in range(N):
        for j in range(N):
            E += P[i][j] * (H[j][i] + F[j][i])
    E = 0.5 * E

    return E