import numpy as np
from scipy.constants import epsilon_0
from scipy.special import j0, j1, k0, k1, kv


def k2(x):
    return kv(2, x)


def eigenvalue_equation(u, w, n_core, n_clad):
    return j1(u) / (u * j0(u)) + (n_clad / n_core) ** 2 * k1(w) / (w * k0(w))


def er_core(r, A, beta, a, u):
    return 1j * beta * (a / u) * A * j1(u / a * r)


def ez_core(r, A, a, u):
    return A * j0(u / a * r) + 0j


def ht_core(r, A, omega, a, n_core, u):
    return 1j * omega * epsilon_0 * n_core ** 2 * (a / u) * A * j1(u / a * r)


def er_clad(r, A, beta, a, u, w):
    return -1j * beta * (a / w) * (j0(u) / k0(w)) * A * k1(w / a * r)


def ez_clad(r, A, a, u, w):
    return (j0(u) / k0(w)) * A * k0(w / a * r) + 0j


def ht_clad(r, A, omega, a, n_clad, u, w):
    return (-1j * omega * epsilon_0 * n_clad ** 2 * (a / w) * (j0(u) / k0(w))
            * A * k1(w / a * r))


def power_core(A, beta, omega, a, n_core, n_clad, u, w):
    return ((np.pi / 2) * omega * epsilon_0 * n_core ** 2 * beta * A ** 2
            * (a ** 4 / u ** 2) * j1(u) ** 2 * (1 + (n_core ** 2 / n_clad ** 2)
            * (w ** 2 / u ** 2) * (k0(w) * k2(w) / k1(w) ** 2)))


def power_clad(A, beta, omega, a, n_core, n_clad, u, w):
    return ((np.pi / 2) * omega * epsilon_0 * n_core ** 2 * beta * A ** 2
            * (a ** 4 / u ** 2) * j1(u) ** 2 * (n_core ** 2 / n_clad ** 2)
            * (w ** 2 / u ** 2) * ((k0(w) * k2(w) / k1(w) ** 2) - 1))


def amplitude(P, beta, omega, a, n_core, n_clad, u, w):
    return (np.sqrt(P / (power_core(1.0, beta, omega, a, n_core, n_clad, u, w)
            + power_clad(1.0, beta, omega, a, n_core, n_clad, u, w))))
