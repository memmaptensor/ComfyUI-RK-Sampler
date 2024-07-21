from typing import Optional

import torch
from torchode.interpolation import ThirdOrderPolynomialInterpolation
from torchode.terms import ODETerm

from .runge_kutta import ButcherTableau, ERKInterpolationData, ExplicitRungeKutta


class FEKutta4(ExplicitRungeKutta):
    ORDER = 4
    NFE_PER_STEP = 4
    TABLEAU = ButcherTableau.from_lists(
        c=[0, 1 / 2, 1 / 2, 1],
        a=[[], [1 / 2], [0, 1 / 2], [0, 0, 1]],
        b=[1 / 6, 1 / 3, 1 / 3, 1 / 6],
    )

    def __init__(self, term: Optional[ODETerm] = None):
        super().__init__(term, self.TABLEAU)

    @torch.jit.export
    def convergence_order(self):
        return self.ORDER

    @torch.jit.export
    def build_interpolation(self, data: ERKInterpolationData):
        return ThirdOrderPolynomialInterpolation.from_k(data.t0, data.dt, data.y0, data.y1, data.k)
