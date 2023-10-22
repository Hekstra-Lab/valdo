import numpy as np


def knn_weight_norm_p01(delta_r):
    r0 = 0.01  # inverse Angstrom
    return (delta_r > 1e-8) * np.exp(-0.5 * (delta_r / r0) ** 2)


def knn_weight_exp_p01(delta_r):
    r0 = 0.01  # inverse Angstrom
    return (delta_r > 1e-8) * np.exp(-delta_r / r0)


def knn_weight_norm_p02(delta_r):
    r0 = 0.02  # inverse Angstrom
    return (delta_r > 1e-8) * np.exp(-0.5 * (delta_r / r0) ** 2)


def knn_weight_exp_p02(delta_r):
    r0 = 0.02  # inverse Angstrom
    return (delta_r > 1e-8) * np.exp(-delta_r / r0)


def knn_weight_norm_p03(delta_r):
    r0 = 0.03  # inverse Angstrom
    return (delta_r > 1e-8) * np.exp(-0.5 * (delta_r / r0) ** 2)


def knn_weight_exp_p03(delta_r):
    r0 = 0.03  # inverse Angstrom
    return (delta_r > 1e-8) * np.exp(-delta_r / r0)


def knn_weight_norm_p05(delta_r):
    r0 = 0.05  # inverse Angstrom
    return (delta_r > 1e-8) * np.exp(-0.5 * (delta_r / r0) ** 2)


def knn_weight_exp_p05(delta_r):
    r0 = 0.05  # inverse Angstrom
    return (delta_r > 1e-8) * np.exp(-delta_r / r0)
