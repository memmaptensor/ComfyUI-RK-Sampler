from typing import Optional

import torch
from torchode.interpolation import ThirdOrderPolynomialInterpolation
from torchode.terms import ODETerm

from .runge_kutta import ButcherTableau, ERKInterpolationData, ExplicitRungeKutta


class FEWray3(ExplicitRungeKutta):
    ORDER = 3
    NFE_PER_STEP = 3
    TABLEAU = ButcherTableau.from_lists(
        c=[0, 8 / 15, 2 / 3],
        a=[[], [8 / 15], [1 / 4, 5 / 12]],
        b=[1 / 4, 0, 3 / 4],
    )

    def __init__(self, term: Optional[ODETerm] = None):
        super().__init__(term, self.TABLEAU)

    @torch.jit.export
    def convergence_order(self):
        return self.ORDER

    @torch.jit.export
    def build_interpolation(self, data: ERKInterpolationData):
        return ThirdOrderPolynomialInterpolation.from_k(data.t0, data.dt, data.y0, data.y1, data.k)
