from typing import Optional

import torch
from torchode.interpolation import ThirdOrderPolynomialInterpolation
from torchode.terms import ODETerm

from .runge_kutta import ButcherTableau, ERKInterpolationData, ExplicitRungeKutta


class AERalston2(ExplicitRungeKutta):
    ORDER = 2
    NFE_PER_STEP = 2
    TABLEAU = ButcherTableau.from_lists(
        c=[0, 2 / 3],
        a=[[], [2 / 3]],
        b=[1 / 4, 3 / 4],
        b_err=[1 / 4 - -2 / 4, 3 / 4 - 6 / 4],
    )

    def __init__(self, term: Optional[ODETerm] = None):
        super().__init__(term, self.TABLEAU)

    @torch.jit.export
    def convergence_order(self):
        return self.ORDER

    @torch.jit.export
    def build_interpolation(self, data: ERKInterpolationData):
        return ThirdOrderPolynomialInterpolation.from_k(data.t0, data.dt, data.y0, data.y1, data.k)
