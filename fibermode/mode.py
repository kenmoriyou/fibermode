from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np
import scipy.optimize as optimize
from numpy.typing import ArrayLike, NDArray
from scipy.constants import c

Field = NDArray[np.complex128]


@dataclass(repr=False, frozen=True)
class Mode(ABC):
    lam: float
    a: float
    n_core: float
    n_clad: float
    n: int
    m: int
    beta: float = field(init=False)
    power: Optional[float] = field(default=None)

    def __repr__(self):
        name = type(self).__name__
        return f"{name}({self.n}, {self.m})"

    @property
    def k(self) -> float:
        return 2 * np.pi / self.lam

    @property
    def omega(self) -> float:
        return c * self.k

    @property
    def V(self) -> float:
        return self.k * self.a * np.sqrt(self.n_core ** 2 - self.n_clad ** 2)

    @property
    def u(self) -> float:
        return self.a * np.sqrt((self.k * self.n_core) ** 2 - self.beta ** 2)

    @property
    def w(self) -> float:
        return self.a * np.sqrt(self.beta ** 2 - (self.k * self.n_clad) ** 2)

    @staticmethod
    def get_broadcast_shape(*args: ArrayLike) -> Tuple[int, ...]:
        shapes = [np.shape(x) for x in args]
        return np.broadcast_shapes(*shapes)

    @staticmethod
    def zero_field(r: ArrayLike,
                   theta: ArrayLike,
                   phi: ArrayLike = 0.0) -> Field:
        shape = np.broadcast_shapes(Mode.get_broadcast_shape(r, theta, phi))
        return np.zeros(shape, dtype=np.complex128)

    @abstractmethod
    def calc_beta(
        self,
        eigenvalue_equation: Callable[[float], float]
    ) -> float:
        # get zero crossings
        u = np.linspace(0, self.V, 500)[1:-1]
        eve = eigenvalue_equation(u)
        eve[np.abs(eve) > self._tol] = np.nan
        zero_crossings = np.where(eve[:-1] * eve[1:] < 0)[0]

        # if the mth zero-crossing doesn't exit, the mth mode cannot be excited
        if len(zero_crossings) < self.m:
            return np.nan

        # solve eve
        i_zc = zero_crossings[self.m - 1]
        bracket = (u[i_zc], u[i_zc + 1])
        u_root = optimize.root_scalar(eigenvalue_equation,
                                      bracket=bracket,
                                      method="bisect").root

        # return beta
        return np.sqrt(self.k ** 2 * self.n_core ** 2 - (u_root / self.a) ** 2)

    # electric field
    @abstractmethod
    def er(self,
           r: ArrayLike,
           theta: ArrayLike,
           phi: ArrayLike = 0.0) -> Field:
        ...

    @abstractmethod
    def et(self,
           r: ArrayLike,
           theta: ArrayLike,
           phi: ArrayLike = 0.0) -> Field:
        ...

    @abstractmethod
    def ez(self,
           r: ArrayLike,
           theta: ArrayLike,
           phi: ArrayLike = 0.0) -> Field:
        ...

    @abstractmethod
    def ex(self,
           r: ArrayLike,
           theta: ArrayLike,
           phi: ArrayLike = 0.0) -> Field:
        return (self.er(r, theta, phi) * np.cos(theta)
                - self.et(r, theta, phi) * np.sin(theta))

    @abstractmethod
    def ey(self,
           r: ArrayLike,
           theta: ArrayLike,
           phi: ArrayLike = 0.0) -> Field:
        return (self.er(r, theta, phi) * np.sin(theta)
                + self.et(r, theta, phi) * np.cos(theta))

    # magnetic field
    @abstractmethod
    def hr(self,
           r: ArrayLike,
           theta: ArrayLike,
           phi: ArrayLike = 0.0) -> Field:
        ...

    @abstractmethod
    def ht(self,
           r: ArrayLike,
           theta: ArrayLike,
           phi: ArrayLike = 0.0) -> Field:
        ...

    @abstractmethod
    def hz(self,
           r: ArrayLike,
           theta: ArrayLike,
           phi: ArrayLike = 0.0) -> Field:
        ...

    @abstractmethod
    def hx(self,
           r: ArrayLike,
           theta: ArrayLike,
           phi: ArrayLike = 0.0) -> Field:
        return (self.hr(r, theta, phi) * np.cos(theta)
                - self.ht(r, theta, phi) * np.sin(theta))

    @abstractmethod
    def hy(self,
           r: ArrayLike,
           theta: ArrayLike,
           phi: ArrayLike = 0.0) -> Field:
        return (self.hr(r, theta, phi) * np.sin(theta)
                + self.ht(r, theta, phi) * np.cos(theta))
