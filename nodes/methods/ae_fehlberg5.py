from typing import Optional

import torch
from torchode.interpolation import ThirdOrderPolynomialInterpolation
from torchode.terms import ODETerm

from .runge_kutta import ButcherTableau, ERKInterpolationData, ExplicitRungeKutta


class AEFehlberg5(ExplicitRungeKutta):
    ORDER = 5
    NFE_PER_STEP = 6
    TABLEAU = ButcherTableau.from_lists(
        c=[0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2],
        a=[
            [],
            [1 / 4],
            [3 / 32, 9 / 32],
            [1932 / 2197, -7200 / 2197, 7296 / 2197],
            [439 / 216, -8, 3680 / 513, -845 / 4104],
            [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40],
        ],
        b=[16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55],
        b_err=[
            16 / 135 - 25 / 216,
            0 - 0,
            6656 / 12825 - 1408 / 2565,
            28561 / 56430 - 2197 / 4104,
            -9 / 50 - -1 / 5,
            2 / 55 - 0,
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
