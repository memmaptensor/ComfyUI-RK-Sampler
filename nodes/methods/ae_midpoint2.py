from typing import Optional

import torch
from torchode.interpolation import ThirdOrderPolynomialInterpolation
from torchode.terms import ODETerm

from .runge_kutta import ButcherTableau, ERKInterpolationData, ExplicitRungeKutta


class AEMidpoint2(ExplicitRungeKutta):
    ORDER = 2
    NFE_PER_STEP = 2
    TABLEAU = ButcherTableau.from_lists(
        c=[0, 1 / 2],
        a=[[], [1 / 2]],
        b=[0, 1],
        b_err=[0 - -1, 1 - 2],
    )

    def __init__(self, term: Optional[ODETerm] = None):
        super().__init__(term, self.TABLEAU)

    @torch.jit.export
    def convergence_order(self):
        return self.ORDER

    @torch.jit.export
    def build_interpolation(self, data: ERKInterpolationData):
        return ThirdOrderPolynomialInterpolation.from_k(data.t0, data.dt, data.y0, data.y1, data.k)
