from typing import Optional

import torch
from torchode.interpolation import FourthOrderPolynomialInterpolation
from torchode.terms import ODETerm

from .runge_kutta import ButcherTableau, ERKInterpolationData, ExplicitRungeKutta


class AEDopri5(ExplicitRungeKutta):
    ORDER = 5
    NFE_PER_STEP = 6
    TABLEAU = ButcherTableau.from_lists(
        c=[0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1],
        a=[
            [],
            [1 / 5],
            [3 / 40, 9 / 40],
            [44 / 45, -56 / 15, 32 / 9],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
        ],
        b=[35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
        b_err=[
            35 / 384 - 5179 / 57600,
            0 - 0,
            500 / 1113 - 7571 / 16695,
            125 / 192 - 393 / 640,
            -2187 / 6784 - -92097 / 339200,
            11 / 84 - 187 / 2100,
            0 - 1 / 40,
        ],
        b_other=[
            [
                6025192743 / 30085553152 / 2,
                0,
                51252292925 / 65400821598 / 2,
                -2691868925 / 45128329728 / 2,
                187940372067 / 1594534317056 / 2,
                -1776094331 / 19743644256 / 2,
                11237099 / 235043384 / 2,
            ]
        ],
    )

    def __init__(self, term: Optional[ODETerm] = None):
        super().__init__(term, self.TABLEAU)

    @torch.jit.export
    def convergence_order(self):
        return self.ORDER

    @torch.jit.export
    def build_interpolation(self, data: ERKInterpolationData):
        return FourthOrderPolynomialInterpolation.from_k(
            data.t0, data.dt, data.y0, data.y1, data.k, data.tableau.b_other[0]
        )
