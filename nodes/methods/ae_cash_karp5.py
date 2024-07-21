from typing import Optional

import torch
from torchode.interpolation import ThirdOrderPolynomialInterpolation
from torchode.terms import ODETerm

from .runge_kutta import ButcherTableau, ERKInterpolationData, ExplicitRungeKutta


class AECashKarp5(ExplicitRungeKutta):
    ORDER = 5
    NFE_PER_STEP = 6
    TABLEAU = ButcherTableau.from_lists(
        c=[0, 1 / 5, 3 / 10, 3 / 5, 1, 7 / 8],
        a=[
            [],
            [1 / 5],
            [3 / 40, 9 / 40],
            [3 / 10, -9 / 10, 6 / 5],
            [-11 / 54, 5 / 2, -70 / 27, 35 / 27],
            [1631 / 55296, 175 / 512, 575 / 13824, 44275 / 110592, 253 / 4096],
        ],
        b=[37 / 378, 0, 250 / 621, 125 / 594, 0, 512 / 1771],
        b_err=[
            37 / 378 - 2825 / 27648,
            0 - 0,
            250 / 621 - 18575 / 48384,
            125 / 594 - 13525 / 55296,
            0 - 277 / 14336,
            512 / 1771 - 1 / 4,
        ],
    )

    def __init__(self, term: Optional[ODETerm] = None):
        super().__init__(term, self.TABLEAU)

    @torch.jit.export
    def convergence_order(self):
        return self.ORDER

    @torch.jit.export
    def build_interpolation(self, data: ERKInterpolationData):
        return ThirdOrderPolynomialInterpolation.from_k(data.t0, data.dt, data.y0, data.y1, data.k)
