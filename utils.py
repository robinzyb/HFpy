"""
some misc utils
do the simple integration
"""
import numpy as np


def get_normalization(alpha):
    """
    get the normalization constant of guassian function
    :param alpha:
    :return:
    """
    return np.power(2*alpha/np.pi, 3/4)

def get_basis_list(system):
    """
    get the basis list from a list of element object
    :param system: a list of element object
    :return:
    """
    basis_list = []
    for element in system:
        basis_list += element.basis_list
    return basis_list

def get_orth_basis(S):
    """
    orthogonalize basis
    :param S:
    :return:
    """

    #S^-1/2
    s, U = np.linalg.eig(S)
    s = np.power(s, -0.5)
    s = np.diag(s)
    iU = np.linalg.inv(U)
    X = np.matmul(U, np.matmul(s, iU))

    return X


def F_0(t):
    from math import erf
    if t < 1e-6 :
        return 1.0 - t/3.0
    else:
        return 0.5*(np.pi/t)**0.5 * erf(t**0.5)