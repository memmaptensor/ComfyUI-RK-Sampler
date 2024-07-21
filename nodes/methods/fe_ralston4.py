from math import sqrt
from typing import Optional

import torch
from torchode.interpolation import ThirdOrderPolynomialInterpolation
from torchode.terms import ODETerm

from .runge_kutta import ButcherTableau, ExplicitRungeKutta, ERKInterpolationData

SQRT5 = sqrt(5)


class FERalston4(ExplicitRungeKutta):
    ORDER = 4
    NFE_PER_STEP = 4
    TABLEAU = ButcherTableau.from_lists(
        c=[0, 2 / 5, (14 - 3 * SQRT5) / 16, 1],
        a=[
            [],
            [2 / 5],
            [(-2889 + 1428 * SQRT5) / 1024, (3785 - 1620 * SQRT5) / 1024],
            [(-3365 + 2094 * SQRT5) / 6040, (-975 - 3046 * SQRT5) / 2552, (467040 + 203968 * SQRT5) / 240845],
        ],
        b=[
            (263 + 24 * SQRT5) / 1812,
            (125 - 1000 * SQRT5) / 3828,
            (3426304 + 1661952 * SQRT5) / 5924787,
            (30 - 4 * SQRT5) / 123,
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
