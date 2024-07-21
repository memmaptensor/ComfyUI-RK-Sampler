from typing import Optional

import torch
from torchode.interpolation import ThirdOrderPolynomialInterpolation
from torchode.terms import ODETerm

from .runge_kutta import ButcherTableau, ERKInterpolationData, ExplicitRungeKutta


class AEBosh3(ExplicitRungeKutta):
    ORDER = 3
    NFE_PER_STEP = 3
    TABLEAU = ButcherTableau.from_lists(
        c=[0, 1 / 2, 3 / 4, 1],
        a=[[], [1 / 2], [0, 3 / 4], [2 / 9, 1 / 3, 4 / 9]],
        b=[2 / 9, 1 / 3, 4 / 9, 0],
        b_err=[2 / 9 - 7 / 24, 1 / 3 - 1 / 4, 4 / 9 - 1 / 3, 0 - 1 / 8],
    )

    def __init__(self, term: Optional[ODETerm] = None):
        super().__init__(term, self.TABLEAU)

    @torch.jit.export
    def convergence_order(self):
        return self.ORDER

    @torch.jit.export
    def build_interpolation(self, data: ERKInterpolationData):
        return ThirdOrderPolynomialInterpolation.from_k(data.t0, data.dt, data.y0, data.y1, data.k)
