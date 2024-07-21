import logging

import torch
import torchode
from tqdm.auto import tqdm

import comfy
import comfy.model_patcher
import comfy.samplers
import comfy.utils

from .methods.ae_bosh3 import AEBosh3
from .methods.ae_cash_karp5 import AECashKarp5
from .methods.ae_dopri5 import AEDopri5
from .methods.ae_dopri8 import AEDopri8
from .methods.ae_fehlberg2 import AEFehlberg2
from .methods.ae_fehlberg5 import AEFehlberg5
from .methods.ae_heun_euler2 import AEHeunEuler2
from .methods.ae_midpoint2 import AEMidpoint2
from .methods.ae_ralston2 import AERalston2
from .methods.ae_tsit5 import AETsit5
from .methods.fe_euler1 import FEEuler1
from .methods.fe_heun3 import FEHeun3
from .methods.fe_kutta3 import FEKutta3
from .methods.fe_kutta4 import FEKutta4
from .methods.fe_kutta_38th4 import FEKutta38th4
from .methods.fe_ralston3 import FERalston3
from .methods.fe_ralston4 import FERalston4
from .methods.fe_ssprk3 import FESSPRK3
from .methods.fe_wray3 import FEWray3
from .step_size_controllers.pid_controller import PIDController
from .step_size_controllers.scheduled_controller import ScheduledController

logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter(f"[ComfyUI-RK-Sampler] %(levelname)s - %(message)s"))
logger.addHandler(sh)

I_INF = 2**31 - 1
F_INF = 1e38
F_EPS = 1e-5
ADAPTIVE_METHODS = {
    "ae_bosh3": AEBosh3,
    "ae_cash_karp5": AECashKarp5,
    "ae_dopri5": AEDopri5,
    "ae_dopri8": AEDopri8,
    "ae_fehlberg2": AEFehlberg2,
    "ae_fehlberg5": AEFehlberg5,
    "ae_heun_euler2": AEHeunEuler2,
    "ae_midpoint2": AEMidpoint2,
    "ae_ralston2": AERalston2,
    "ae_tsit5": AETsit5,
}
FIXED_METHODS = {
    "fe_euler1": FEEuler1,
    "fe_heun3": FEHeun3,
    "fe_kutta_38th4": FEKutta38th4,
    "fe_kutta3": FEKutta3,
    "fe_kutta4": FEKutta4,
    "fe_ralston3": FERalston3,
    "fe_ralston4": FERalston4,
    "fe_ssprk3": FESSPRK3,
    "fe_wray3": FEWray3,
}
METHODS = {**ADAPTIVE_METHODS, **FIXED_METHODS}
STEP_SIZE_CONTROLLERS = dict.fromkeys(
    [
        "adaptive_pid",
        "fixed_scheduled",
    ]
)
NORMS = {
    "rms_norm": torchode.step_size_controllers.rms_norm,
    "max_norm": torchode.step_size_controllers.max_norm,
}


class ODETerm:
    def __init__(
        self,
        model,
        x_dtype,
        x_shape,
        t_dtype,
        min_sigma,
        t_max,
        t_min,
        n_steps,
        is_adaptive,
        method,
        extra_args=None,
        callback=None,
    ):
        self.model = model
        self.x_dtype = x_dtype
        self.x_shape = x_shape
        self.t_dtype = t_dtype
        self.min_sigma = min_sigma
        self.t_max = t_max
        self.t_min = t_min
        self.n_steps = n_steps
        self.is_adaptive = is_adaptive
        self.method = method
        self.extra_args = {} if extra_args is None else extra_args
        self.callback = callback
        self.step = 0
        self.nfe_step = 0

        if is_adaptive:
            self.progress_bar = tqdm(total=100, desc=f"Adaptive {method}", unit="%")
        else:
            self.progress_bar = tqdm(total=n_steps, desc=f"Fixed {method}", unit="step")

    def _callback(self, t, y, denoised, mask):
        if self.is_adaptive:
            progress = ((self.t_max - t) / (self.t_max - self.t_min)).detach().mean().item()
            d_progress = progress * 100
            self.progress_bar.update(d_progress - self.step)
            self.step = d_progress
            i = round(progress * self.n_steps)
        else:
            self.progress_bar.update(1)
            self.step += 1
            i = self.step

        if self.callback is not None:
            samples = torch.where(
                mask.view(*mask.shape, 1, 1, 1),
                y,
                denoised,
            )
            self.callback(
                {
                    "x": y.to(self.x_dtype),
                    "i": i - 1,
                    "sigma": t.to(self.t_dtype),
                    "sigma_hat": t.to(self.t_dtype),
                    "denoised": samples.to(self.x_dtype),
                }
            )

    def __call__(self, t, y):
        mask = t <= self.min_sigma
        y = y.reshape(self.x_shape)
        denoised = torch.zeros_like(y)
        if not mask.all():
            denoised[~mask] = self.model(y[~mask], t[~mask], **self.extra_args)
        d = torch.where(
            mask.view(*mask.shape, 1, 1, 1),
            torch.zeros_like(y),
            (y - denoised) / t.view(*t.shape, 1, 1, 1),
        )

        self.nfe_step += 1
        if self.nfe_step % METHODS[self.method].NFE_PER_STEP == 0:
            self._callback(t, y, denoised, mask)

        return d.flatten(start_dim=1)


class RungeKuttaSamplerImpl:
    def __init__(
        self,
        method,
        step_size_controller,
        log_absolute_tolerance,
        log_relative_tolerance,
        pcoeff,
        icoeff,
        dcoeff,
        norm,
        enable_dt_min,
        enable_dt_max,
        dt_min,
        dt_max,
        safety,
        factor_min,
        factor_max,
        max_steps,
        min_sigma,
    ):
        self.method = method
        self.step_size_controller = step_size_controller
        self.atol = 10**log_absolute_tolerance
        self.rtol = 10**log_relative_tolerance
        assert self.atol <= self.rtol
        self.pcoeff = pcoeff
        self.icoeff = icoeff
        self.dcoeff = dcoeff
        self.norm = norm
        self.enable_dt_min = enable_dt_min
        self.enable_dt_max = enable_dt_max
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.safety = safety
        self.factor_min = factor_min
        self.factor_max = factor_max
        self.max_steps = max_steps
        self.min_sigma = min_sigma

    @torch.no_grad()
    def __call__(self, model, x: torch.Tensor, sigmas: torch.Tensor, extra_args=None, callback=None, disable=None):
        dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        t_max = sigmas.max()
        t_min = sigmas.min()
        n_steps = len(sigmas) - 1
        is_adaptive = self.step_size_controller.startswith("adaptive")

        if is_adaptive and (self.method in FIXED_METHODS):
            raise ValueError("Fixed step methods must be used with fixed step size controllers")

        term = torchode.ODETerm(
            ODETerm(
                model=model,
                x_dtype=x.dtype,
                x_shape=x.shape,
                t_dtype=sigmas.dtype,
                min_sigma=self.min_sigma if is_adaptive else 0.0,
                t_max=t_max,
                t_min=t_min,
                n_steps=n_steps,
                is_adaptive=is_adaptive,
                method=self.method,
                extra_args=extra_args,
                callback=callback,
            )
        )

        if self.step_size_controller == "fixed_scheduled":
            step_size_controller = ScheduledController(sigmas=sigmas)
        elif self.step_size_controller == "adaptive_pid":
            step_size_controller = PIDController(
                atol=self.atol,
                rtol=self.rtol,
                pcoeff=self.pcoeff,
                icoeff=self.icoeff,
                dcoeff=self.dcoeff,
                term=term,
                norm=NORMS[self.norm],
                dt_min=self.dt_min if self.enable_dt_min else None,
                dt_max=self.dt_max if self.enable_dt_max else None,
                safety=self.safety,
                factor_min=self.factor_min,
                factor_max=self.factor_max,
            )

        step_method = METHODS[self.method](term=term)
        adjoint = torchode.AutoDiffAdjoint(step_method, step_size_controller, max_steps=self.max_steps)
        problem = torchode.InitialValueProblem(
            y0=x.flatten(start_dim=1).to(dtype),
            t_start=torch.full((x.shape[0],), t_max, dtype=dtype, device=sigmas.device),
            t_end=torch.full((x.shape[0],), t_min, dtype=dtype, device=sigmas.device),
        )
        result = adjoint.solve(problem)
        samples = result.ys[:, -1].reshape(x.shape).to(x.dtype)

        success = True
        for i, status in enumerate(result.status):
            status = status.item()
            if status != 0:
                success = False
                samples[i] = torch.full_like(samples[i], torch.nan)
                reason = torchode.status_codes.Status(status)
                logger.warning(f"Sample #{i} failed with reason: {reason}")

        if success:
            term.f.progress_bar.update(term.f.progress_bar.total - term.f.progress_bar.n)

        if callback is not None:
            callback(
                {
                    "x": samples,
                    "i": n_steps - 1,
                    "sigma": t_min,
                    "sigma_hat": t_min,
                    "denoised": samples,
                }
            )

        return samples


class RungeKuttaSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "method": (list(METHODS.keys()), {"default": "ae_bosh3"}),
                "step_size_controller": (list(STEP_SIZE_CONTROLLERS.keys()), {"default": "adaptive_pid"}),
                "log_absolute_tolerance": (
                    "FLOAT",
                    {"default": -3.5, "min": -F_INF, "max": F_INF, "step": F_EPS, "round": False},
                ),
                "log_relative_tolerance": (
                    "FLOAT",
                    {"default": -2.5, "min": -F_INF, "max": F_INF, "step": F_EPS, "round": False},
                ),
                "pcoeff": ("FLOAT", {"default": 0.0, "min": 0, "max": F_INF, "step": F_EPS, "round": False}),
                "icoeff": ("FLOAT", {"default": 1.0, "min": 0, "max": F_INF, "step": F_EPS, "round": False}),
                "dcoeff": ("FLOAT", {"default": 0.0, "min": 0, "max": F_INF, "step": F_EPS, "round": False}),
                "norm": (list(NORMS.keys()), {"default": "rms_norm"}),
                "enable_dt_min": ("BOOLEAN", {"default": False}),
                "enable_dt_max": ("BOOLEAN", {"default": True}),
                "dt_min": ("FLOAT", {"default": -1.0, "min": -F_INF, "max": F_INF, "step": F_EPS, "round": False}),
                "dt_max": ("FLOAT", {"default": 0.0, "min": -F_INF, "max": F_INF, "step": F_EPS, "round": False}),
                "safety": ("FLOAT", {"default": 0.9, "min": 0, "max": F_INF, "step": F_EPS, "round": False}),
                "factor_min": ("FLOAT", {"default": 0.2, "min": 0, "max": F_INF, "step": F_EPS, "round": False}),
                "factor_max": ("FLOAT", {"default": 10, "min": 0, "max": F_INF, "step": F_EPS, "round": False}),
                "max_steps": ("INT", {"default": I_INF, "min": 0, "max": I_INF, "step": 1, "round": False}),
                "min_sigma": ("FLOAT", {"default": 1e-5, "min": 0, "max": F_INF, "step": F_EPS, "round": False}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "sampling/custom_sampling/samplers"

    def get_sampler(self, **kwargs):
        return (comfy.samplers.KSAMPLER(RungeKuttaSamplerImpl(**kwargs)),)


NODE_CLASS_MAPPINGS = {
    "RungeKuttaSampler": RungeKuttaSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RungeKuttaSampler": "Runge-Kutta Sampler",
}
