from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torchode import status_codes
from torchode.problems import InitialValueProblem
from torchode.single_step_methods import SingleStepMethod
from torchode.solution import Solution
from torchode.step_size_controllers import StepSizeController
from torchode.terms import ODETerm
from torchode.typing import *


class AutoDiffAdjoint(nn.Module):
    def __init__(
        self,
        step_method: SingleStepMethod,
        step_size_controller: StepSizeController,
        *,
        max_steps: Optional[int] = None,
        backprop_through_step_size_control: bool = True,
        dense_output: bool = True,
    ):
        super().__init__()

        self.step_method = step_method
        self.step_size_controller = step_size_controller
        self.max_steps = max_steps
        self.backprop_through_step_size_control = backprop_through_step_size_control
        self.dense_output = dense_output

    @torch.jit.export
    def solve(
        self,
        problem: InitialValueProblem,
        term: Optional[ODETerm] = None,
        dt0: Optional[TimeTensor] = None,
        args: Any = None,
    ) -> Solution:
        step_method, step_size_controller = self.step_method, self.step_size_controller
        device, batch_size = problem.device, problem.batch_size
        t_start, t_end, t_eval = problem.t_start, problem.t_end, problem.t_eval
        time_direction = problem.time_direction.to(dtype=t_start.dtype)

        ###############################
        # Initialize the solver state #
        ###############################

        t = t_start
        y = problem.y0
        stats_n_steps = y.new_zeros(batch_size, dtype=torch.long)
        stats_n_accepted = y.new_zeros(batch_size, dtype=torch.long)
        stats: Dict[str, Any] = {}

        # Compute the boundaries in time to ensure that we never step outside of them
        t_min = torch.minimum(t_start, t_end)
        t_max = torch.maximum(t_end, t_start)

        # TorchScript is not smart enough yet to figure out that we only access these
        # variables when they have been defined, so we have to always define them and
        # intialize them to any valid tensor.
        y_eval: torch.Tensor = y
        not_yet_evaluated: torch.Tensor = y
        minus_t_eval_normalized: torch.Tensor = y

        if t_eval is not None:
            y_eval = y.new_empty((batch_size, problem.n_evaluation_points, problem.n_features))

            # Keep track of which evaluation points have not yet been handled
            not_yet_evaluated = torch.ones_like(t_eval, dtype=torch.bool)

        # Normalize the time direction of the evaluation and end times for faster
        # comparisons
        minus_t_end_normalized = -time_direction * t_end
        if t_eval is not None:
            minus_t_eval_normalized = -time_direction[:, None] * t_eval

        # Keep track of which solves are still running
        running = y.new_ones(batch_size, dtype=torch.bool)

        # Initialize additional statistics to track for the integration term
        term_ = term
        if torch.jit.is_scripting() or term is None:
            assert term is None, "The integration term is fixed for JIT compilation"
            term_ = self.step_method.term
        assert term_ is not None
        term_.init(problem, stats)

        # Compute an initial step size
        convergence_order = step_method.convergence_order()
        dt, controller_state, f0 = step_size_controller.init(
            term, problem, convergence_order, dt0, stats=stats, args=args
        )
        method_state = step_method.init(term, problem, f0, stats=stats, args=args)

        # Ensure that the initial dt does not step outside of the time domain
        dt = torch.clamp(dt, t_min - t, t_max - t)

        # TorchScript does not support set_grad_enabled, so we detach manually
        if not self.backprop_through_step_size_control:
            dt = dt.detach()

        ##############################################
        # Take care of evaluation exactly at t_start #
        ##############################################

        # We copy the initial state into the evaluation if the first evaluation point
        # happens to be exactly `t_start`. This is required so that we can later assume
        # that rejection of the step (and therefore also no change in `t`) means that we
        # also did not pass any evaluation points.
        if t_eval is not None:
            eval_at_start = t_eval[:, 0] == t_start
            y_eval[eval_at_start, 0] = y[eval_at_start]
            not_yet_evaluated[eval_at_start, 0] = False

        ####################################
        # Solve the initial value problems #
        ####################################

        # Iterate the single step method until all ODEs have been solved up to their end
        # point or any of them failed
        max_steps = self.max_steps
        while True:
            step_out = step_method.step(term, running, y, t, dt, method_state, stats=stats, args=args)
            step_result, interp_data, method_state_next, method_status = step_out
            controller_out = step_size_controller.adapt_step_size(t, dt, y, step_result, controller_state, stats)
            accept, dt_next, controller_state_next, controller_status = controller_out

            # TorchScript does not support set_grad_enabled, so we detach manually
            if not self.backprop_through_step_size_control:
                dt_next = dt_next.detach()

            # Update the solver state where the step was accepted
            to_update = accept & running
            t = torch.where(to_update, t + dt, t)
            y = torch.where(to_update[:, None], step_result.y, y)
            method_state = step_method.merge_states(to_update, method_state_next, method_state)

            #####################
            # Update statistics #
            #####################

            stats_n_steps.add_(running)
            stats_n_accepted.add_(to_update)

            ##################################
            # Update solver state and status #
            ##################################

            # Stop a solve if `t` has passed its endpoint in the direction of time
            running = torch.addcmul(minus_t_end_normalized, time_direction, t) < 0.0

            status = method_status
            if status is None:
                status = controller_status
            elif controller_status is not None:
                status = torch.maximum(status, controller_status)
            if max_steps is not None:
                status = torch.where(
                    stats_n_steps >= max_steps,
                    status_codes.REACHED_MAX_STEPS,
                    status if status is not None else status_codes.SUCCESS,
                )

            # We evaluate the termination condition here already and initiate a
            # non-blocking transfer to the CPU to increase the chance that we won't have
            # to wait for the result when we actually check the termination condition
            continue_iterating = torch.any(running)
            if status is not None:
                continue_iterating = continue_iterating & torch.all(status == status_codes.SUCCESS)
            continue_iterating = continue_iterating.to("cpu", non_blocking=True)

            # There is a bug as of pytorch 1.12.1 where non-blocking transfer from
            # device to host can sometimes gives the wrong result, so we place this
            # event after the transfer to ensure that the transfer has actually happened
            # by the time we evaluate the result.
            if device.type == "cuda":
                continue_iterating_done = torch.cuda.Event()
                continue_iterating_done.record(torch.cuda.current_stream(device))
            else:
                continue_iterating_done = None

            #########################
            # Evaluate the solution #
            #########################

            # Evaluate the solution at all evaluation points that have been passed in
            # this step.
            #
            # We always build the interpolation and evaluate it, even if no evaluation
            # points have actually been passed, because this avoids a CPU-GPU
            # synchronization and for time series models we expect that most steps will
            # pass at least one evaluation point across the whole batch (usually more).
            if t_eval is not None:
                to_be_evaluated = (
                    torch.addcmul(
                        minus_t_eval_normalized,
                        time_direction[:, None],
                        t[:, None],
                    )
                    >= 0.0
                ) & not_yet_evaluated
                if to_be_evaluated.any():
                    interpolation = step_method.build_interpolation(interp_data)
                    nonzero = to_be_evaluated.nonzero()
                    sample_idx, eval_t_idx = nonzero[:, 0], nonzero[:, 1]
                    y_eval[sample_idx, eval_t_idx] = interpolation.evaluate(t_eval[sample_idx, eval_t_idx], sample_idx)

                    not_yet_evaluated = torch.logical_xor(to_be_evaluated, not_yet_evaluated)

            ########################
            # Update the step size #
            ########################

            # We update the step size and controller state only for solves which will
            # still be running in the next iteration. Otherwise, a finished instance
            # with an adaptive step size controller could reach infinite step size if
            # its final error was small and another instance is running for many steps.
            # This would then cancel the solve even though the "problematic" instance is
            # not even supposed to be running anymore.

            dt = torch.where(running, dt_next, dt)

            # Ensure that we do not step outside of the time domain, even for instances
            # that are not running anymore
            dt = torch.clamp(dt, t_min - t, t_max - t)

            controller_state = step_size_controller.merge_states(running, controller_state_next, controller_state)

            if continue_iterating_done is not None:
                continue_iterating_done.synchronize()

            step_method.term.f.trigger_callback(t, y)

            if continue_iterating:
                continue

            ##################################################
            # Finalize the solver and construct the solution #
            ##################################################

            # Ensure that the user always gets a status tensor
            if status is None:
                status = torch.tensor(status_codes.SUCCESS, dtype=torch.long, device=device).expand(batch_size)

            # Put the step statistics into the stats dict in the end, so that
            # we don't have to type-assert all the time in torchscript
            stats["n_steps"] = stats_n_steps
            stats["n_accepted"] = stats_n_accepted

            # The finalization scope is in the scope of the while loop so that the
            # `t_eval is None` case can access the `interp_data` in TorchScript.
            # Declaring `interp_data` outside of the loop does not work because its type
            # depends on the step method.

            if t_eval is not None:
                # Report the number of evaluation steps that have been initialized with
                # actual data at termination. Depending on the termination condition,
                # the data might be NaN or inf but it will not be uninitialized memory.
                #
                # As of torch 1.12.1, searchsorted is not implemented for bool tensors,
                # so we convert to int first.
                stats["n_initialized"] = torch.searchsorted(
                    not_yet_evaluated.int(),
                    torch.ones((batch_size, 1), dtype=torch.int, device=device),
                ).squeeze(dim=1)

                return Solution(ts=t_eval, ys=y_eval, stats=stats, status=status)
            else:
                # Evaluate the solution only at the end point, e.g. in continuous
                # normalizing flows
                if self.dense_output:
                    interpolation = step_method.build_interpolation(interp_data)
                    y_end = interpolation.evaluate(t_end, torch.arange(batch_size, device=device))
                    stats["n_initialized"] = torch.ones(batch_size, dtype=torch.long, device=device)
                    return Solution(ts=t_end[:, None], ys=y_end[:, None], stats=stats, status=status)
                else:
                    stats["n_initialized"] = torch.ones(batch_size, dtype=torch.long, device=device)
                    return Solution(ts=t_end[:, None], ys=y[:, None], stats=stats, status=status)

        assert False, "unreachable"

    def __repr__(self):
        return (
            f"AutoDiffAdjoint(step_method={self.step_method}, "
            f"step_size_controller={self.step_size_controller}, "
            f"max_steps={self.max_steps}, "
            f"backprop_through_step_size_control={self.backprop_through_step_size_control})"
            f"dense_output={self.dense_output}"
        )
