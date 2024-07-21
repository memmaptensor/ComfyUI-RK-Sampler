from typing import Optional

import torch
from torchode.interpolation import LinearInterpolation
from torchode.terms import ODETerm

from .runge_kutta import ButcherTableau, ERKInterpolationData, ExplicitRungeKutta


class FEEuler1(ExplicitRungeKutta):
    ORDER = 1
    NFE_PER_STEP = 1
    TABLEAU = ButcherTableau.from_lists(
        c=[0],
        a=[[]],
        b=[1],
    )

    def __init__(self, term: Optional[ODETerm] = None):
        super().__init__(term, self.TABLEAU)

    @torch.jit.export
    def convergence_order(self):
        return self.ORDER

    @torch.jit.export
    def build_interpolation(self, data: ERKInterpolationData):
        return LinearInterpolation(data.t0, data.dt, data.y0, data.y1)
