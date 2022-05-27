import numpy as np
from scipy.constants import mu_0
from scipy.special import j0, j1, k0, k1, kv


def k2(x):
    return kv(2, x)


def eigenvalue_equation(u, w):
    return j1(u) / (u * j0(u)) + k1(w) / (w * k0(w))


def et_core(r, A, omega, a, u):
    return -1j * omega * mu_0 * (a / u) * A * j1(u / a * r)


def hr_core(r, A, beta, a, u):
    return 1j * beta * (a / u) * A * j1(u / a * r)


def hz_core(r, A, a, u):
    return A * j0(u / a * r) + 0j


def et_clad(r, A, omega, a, u, w):
    return 1j * omega * mu_0 * (a / w) * (j0(u) / k0(w)) * A * k1(w / a * r)


def hr_clad(r, A, beta, a, u, w):
    return -1j * beta * (a / w) * (j0(u) / k0(w)) * A * k1(w / a * r)


def hz_clad(r, A, a, u, w):
    return (j0(u) / k0(w)) * A * k0(w / a * r) + 0j


def power_core(A, beta, omega, a, u, w):
    return ((np.pi / 2) * omega * mu_0 * beta * A ** 2 * (a ** 4 / u ** 2)
            * j1(u) ** 2 * (1 + (w ** 2 / u ** 2)
            * (k0(w) * k2(w) / k1(w) ** 2)))


def power_clad(A, beta, omega, a, u, w):
    return ((np.pi / 2) * omega * mu_0 * beta * A ** 2 * (a ** 4 / u ** 2)
            * j1(u) ** 2 * ((k0(w) * k2(w) / k1(w) ** 2) - 1))


def amplitude(P, beta, omega, a, n_core, n_clad, u, w):
    return (np.sqrt(P / (power_core(1.0, beta, omega, a, n_core, n_clad, u, w)
            + power_clad(1.0, beta, omega, a, n_core, n_clad, u, w))))
