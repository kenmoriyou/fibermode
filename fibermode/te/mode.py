from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike
import scipy.optimize as optimize

import fibermode.te.equations as eq
from fibermode.mode import Field, Mode


@dataclass(frozen=True, repr=False)
class TE(Mode):
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
        assert self.m >= 1

        # set beta
        beta = self.calc_beta(lambda u: eq.eigenvalue_equation(
            u, np.sqrt(self.V ** 2 - u ** 2)
        ))
        object.__setattr__(self, "beta", beta)

    def calc_beta(
        self,
        eigenvalue_equation: Callable[[float], float]
    ) -> float:
        return super().calc_beta(eigenvalue_equation)

    def er(self,
           r: ArrayLike,
           theta: ArrayLike,
           phi: ArrayLike = 0.0) -> Field:
        return self.zero_field(r, theta, phi)

    def et(self,
           r: ArrayLike,
           theta: ArrayLike,
           phi: ArrayLike = 0.0) -> Field:
        if self.beta is np.nan:
            return self.zero_field(r, theta, phi)

        field = np.where(
            r > self.a,
            eq.et_clad(r, self.A, self.omega, self.a, self.u, self.w),
            eq.et_core(r, self.A, self.omega, self.a, self.u)
        )
        shape = self.get_broadcast_shape(r, theta, phi)
        return np.broadcast_to(field, shape)

    def ez(self,
           r: ArrayLike,
           theta: ArrayLike,
           phi: ArrayLike = 0.0) -> Field:
        return self.zero_field(r, theta, phi)

    def ex(self,
           r: ArrayLike,
           theta: ArrayLike,
           phi: ArrayLike = 0.0) -> Field:
        if self.beta is np.nan:
            return self.zero_field(r, theta, phi)

        return -self.et(r, theta, phi) * np.sin(theta)

    def ey(self,
           r: ArrayLike,
           theta: ArrayLike,
           phi: ArrayLike = 0.0) -> Field:
        if self.beta is np.nan:
            return self.zero_field(r, theta, phi)

        return self.et(r, theta, phi) * np.cos(theta)

    def hr(self,
           r: ArrayLike,
           theta: ArrayLike,
           phi: ArrayLike = 0.0) -> Field:
        if self.beta is np.nan:
            return self.zero_field(r, theta, phi)

        field = np.where(
            r > self.a,
            eq.hr_clad(r, self.A, self.beta, self.a, self.u, self.w),
            eq.hr_core(r, self.A, self.beta, self.a, self.u)
        )
        shape = self.get_broadcast_shape(r, theta, phi)
        return np.broadcast_to(field, shape)

    def ht(self,
           r: ArrayLike,
           theta: ArrayLike,
           phi: ArrayLike = 0.0) -> Field:
        return self.zero_field(r, theta, phi)

    def hz(self,
           r: ArrayLike,
           theta: ArrayLike,
           phi: ArrayLike = 0.0) -> Field:
        if self.beta is np.nan:
            return self.zero_field(r, theta, phi)

        field = np.where(
            r > self.a,
            eq.hz_clad(r, self.A, self.a, self.u, self.w),
            eq.hz_core(r, self.A, self.a, self.u)
        )
        shape = self.get_broadcast_shape(r, theta, phi)
        return np.broadcast_to(field, shape)

    def hx(self,
           r: ArrayLike,
           theta: ArrayLike,
           phi: ArrayLike = 0.0) -> Field:
        if self.beta is np.nan:
            return self.zero_field(r, theta, phi)

        return self.hr(r, theta, phi) * np.cos(theta)

    def hy(self,
           r: ArrayLike,
           theta: ArrayLike,
           phi: ArrayLike = 0.0) -> Field:
        if self.beta is np.nan:
            return self.zero_field(r, theta, phi)

        return self.hr(r, theta, phi) * np.sin(theta)
