from typing import Any, Dict, Optional, Tuple

import torch
from torchode.problems import InitialValueProblem
from torchode.single_step_methods import StepResult
from torchode.step_size_controllers import StepSizeController
from torchode.terms import ODETerm
from torchode.typing import *


class ScheduledState:
    def __init__(self, accept_all: AcceptTensor, step: int):
        self.accept_all = accept_all
        self.step = step


class ScheduledController(StepSizeController[ScheduledState]):
    def __init__(self, sigmas: torch.Tensor):
        super().__init__()
        self.dt = [sigmas[i + 1] - sigmas[i] for i in range(len(sigmas) - 1)]

    @torch.jit.export
    def init(
        self,
        term: Optional[ODETerm],
        problem: InitialValueProblem,
        method_order: int,
        dt0: Optional[TimeTensor],
        *,
        stats: Dict[str, Any],
        args: Any,
    ):
        assert dt0 is None
        return (
            self.dt[0],
            ScheduledState(accept_all=torch.ones(problem.batch_size, device=problem.device, dtype=torch.bool), step=0),
            None,
        )

    @torch.jit.export
    def adapt_step_size(
        self,
        t0: TimeTensor,
        dt: TimeTensor,
        y0: DataTensor,
        step_result: StepResult,
        state: ScheduledState,
        stats: Dict[str, Any],
    ) -> Tuple[AcceptTensor, TimeTensor, ScheduledState, Optional[StatusTensor]]:
        state.step += 1

        if state.step >= len(self.dt) - 1:
            dt_next = -t0
        else:
            dt_next = self.dt[state.step]

        return state.accept_all, dt_next, state, None

    @torch.jit.export
    def merge_states(self, running: AcceptTensor, current: ScheduledState, previous: ScheduledState) -> ScheduledState:
        return current
