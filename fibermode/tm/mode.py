from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike

import fibermode.tm.equations as eq
from fibermode.mode import Mode


@dataclass(frozen=True, repr=False)
class TM(Mode):
    lam: float
    a: float
    n_core: float
    n_clad: float
    n: int = 0
    m: int = field(default=1)
    beta: float = field(init=False)
    power: Optional[float] = field(default=None)
    _tol: float = 1.0

    def __post_init__(self) -> None:
        # set beta
        beta = self.calc_beta(lambda u: eq.eigenvalue_equation(
            u, np.sqrt(self.V ** 2 - u ** 2)), self.n_core, self.n_clad
        )
        object.__setattr__(self, "beta", beta)

    def calc_beta(
        self,
        eigenvalue_equation: Callable[[float], float]
    ) -> float:
        return super().calc_beta(eigenvalue_equation)

    def er(self,
           r: ArrayLike,
           theta: ArrayLike,
           phi: ArrayLike = 0.0) -> ArrayLike:
        if self.beta is np.nan:
            return self.zero_field(r, theta, phi)

        field = np.where(
            r > self.a,
            eq.er_clad(r, self.A, self.beta, self.a, self.u, self.w),
            eq.er_core(r, self.A, self.beta, self.a, self.u)
        )
        shape = self.get_broadcast_shape(r, theta, phi)
        return np.broadcast_to(field, shape)

    def et(self,
           r: ArrayLike,
           theta: ArrayLike,
           phi: ArrayLike = 0.0) -> ArrayLike:
        return self.zero_field(r, theta, phi)

    def ez(self,
           r: ArrayLike,
           theta: ArrayLike,
           phi: ArrayLike = 0.0) -> ArrayLike:
        if self.beta is np.nan:
            return self.zero_field(r, theta, phi)

        field = np.where(
            r > self.a,
            eq.ez_clad(r, self.A, self.a, self.u, self.w),
            eq.ez_core(r, self.A, self.a, self.u)
        )
        shape = self.get_broadcast_shape(r, theta, phi)
        return np.broadcast_to(field, shape)

    def ex(self,
           r: ArrayLike,
           theta: ArrayLike,
           phi: ArrayLike = 0.0) -> ArrayLike:
        if self.beta is np.nan:
            return self.zero_field(r, theta, phi)

        return self.er(r, theta, phi) * np.cos(theta)

    def ey(self,
           r: ArrayLike,
           theta: ArrayLike,
           phi: ArrayLike = 0.0) -> ArrayLike:
        if self.beta is np.nan:
            return self.zero_field(r, theta, phi)

        return self.er(r, theta, phi) * np.sin(theta)

    def hr(self,
           r: ArrayLike,
           theta: ArrayLike,
           phi: ArrayLike = 0.0) -> ArrayLike:
        return self.zero_field(r, theta, phi)

    def ht(self,
           r: ArrayLike,
           theta: ArrayLike,
           phi: ArrayLike = 0.0) -> ArrayLike:
        if self.beta is np.nan:
            return self.zero_field(r, theta, phi)

        field = np.where(
            r > self.a,
            eq.ht_clad(r, self.A, self.omega, self.a,
                       self.n_clad, self.u, self.w),
            eq.ht_core(r, self.A, self.omega, self.a, self.n_core, self.u)
        )
        shape = self.get_broadcast_shape(r, theta, phi)
        return np.broadcast_to(field, shape)

    def hz(self,
           r: ArrayLike,
           theta: ArrayLike,
           phi: ArrayLike = 0.0) -> ArrayLike:
        return self.zero_field(r, theta, phi)

    def hx(self,
           r: ArrayLike,
           theta: ArrayLike,
           phi: ArrayLike = 0.0) -> ArrayLike:
        if self.beta is np.nan:
            return self.zero_field(r, theta, phi)

        return -self.ht(r, theta, phi) * np.sin(theta)

    def hy(self,
           r: ArrayLike,
           theta: ArrayLike,
           phi: ArrayLike = 0.0) -> ArrayLike:
        if self.beta is np.nan:
            return self.zero_field(r, theta, phi)

        return self.ht(r, theta, phi) * np.cos(theta)
