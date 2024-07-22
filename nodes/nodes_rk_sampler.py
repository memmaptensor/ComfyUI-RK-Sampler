import logging

import numpy as np
import scipy.integrate
import torch
import torchode

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
from .ode_terms.scipy_ode_term import SciPyODETerm
from .ode_terms.torchode_ode_term import TorchODEODETerm
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
ADAPTIVE_SCIPY_METHODS = {
    "se_RK23": {"ORDER": 3, "NFE_PER_STEP": 3},
    "se_RK45": {"ORDER": 5, "NFE_PER_STEP": 6},
    "se_DOP853": {"ORDER": 8, "NFE_PER_STEP": 13},
}
METHODS = {**ADAPTIVE_METHODS, **FIXED_METHODS, **ADAPTIVE_SCIPY_METHODS}
STEP_SIZE_CONTROLLERS = dict.fromkeys(
    [
        "adaptive_pid",
        "fixed_scheduled",
        "adaptive_scipy",
    ]
)
NORMS = {
    "rms_norm": torchode.step_size_controllers.rms_norm,
    "max_norm": torchode.step_size_controllers.max_norm,
}


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
        if log_absolute_tolerance > log_relative_tolerance:
            raise ValueError("log_absolute_tolerance must be less than or equal to log_relative_tolerance")
        self.method = method
        self.step_size_controller = step_size_controller
        self.atol = 10**log_absolute_tolerance
        self.rtol = 10**log_relative_tolerance
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
    def _call_torchode(
        self, model, x: torch.Tensor, sigmas: torch.Tensor, extra_args=None, callback=None, disable=None
    ):
        c_device = "cpu"
        c_dtype = torch.float64
        o_device = x.device
        o_dtype = x.dtype
        o_shape = x.shape
        x = x.detach().to(c_device, dtype=c_dtype)
        sigmas = sigmas.detach().to(c_device, dtype=c_dtype)
        t_max = sigmas.max().item()
        t_min = sigmas.min().item()
        n_steps = len(sigmas) - 1

        term = torchode.ODETerm(
            TorchODEODETerm(
                model=model,
                c_device=c_device,
                c_dtype=c_dtype,
                o_device=o_device,
                o_dtype=o_dtype,
                o_shape=o_shape,
                min_sigma=self.min_sigma,
                t_max=t_max,
                t_min=t_min,
                n_steps=n_steps,
                step_size_controller=self.step_size_controller,
                method=self.method,
                nfe_per_step=METHODS[self.method].NFE_PER_STEP,
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
            y0=x.flatten(start_dim=1),
            t_start=torch.full((o_shape[0],), t_max, device=c_device, dtype=c_dtype),
            t_end=torch.full((o_shape[0],), t_min, device=c_device, dtype=c_dtype),
            t_eval=torch.full((o_shape[0], 1), t_min, device=c_device, dtype=c_dtype),
        )
        result = adjoint.solve(problem)
        samples = result.ys[:, -1].reshape(o_shape).to(o_device, dtype=o_dtype)

        faults = []
        for i, status in enumerate(result.status):
            status = status.item()
            if status != 0:
                samples[i] = torch.full_like(samples[i], torch.nan)
                reason = torchode.status_codes.Status(status)
                faults.append(f"Sample #{i} failed with reason: {reason}")

        if len(faults) == 0:
            term.f.progress_bar.update(term.f.progress_bar.total - term.f.progress_bar.n)
            term.f.progress_bar.close()
        else:
            term.f.progress_bar.close()
            for fault in faults:
                logger.warning(fault)

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

    @torch.no_grad()
    def _call_scipy(self, model, x: torch.Tensor, sigmas: torch.Tensor, extra_args=None, callback=None, disable=None):
        c_device = "cpu"
        c_dtype = np.float64
        o_device = x.device
        o_dtype = x.dtype
        o_shape = x.shape
        x: np.ndarray = x.detach().to(c_device).numpy().astype(c_dtype)
        sigmas: np.ndarray = sigmas.detach().to(c_device).numpy().astype(c_dtype)
        t_max = sigmas.max()
        t_min = sigmas.min()
        n_steps = len(sigmas) - 1

        samples = []
        for i in range(o_shape[0]):
            term = SciPyODETerm(
                model=model,
                c_device=c_device,
                c_dtype=c_dtype,
                o_device=o_device,
                o_dtype=o_dtype,
                o_shape=o_shape,
                min_sigma=self.min_sigma,
                t_max=t_max,
                t_min=t_min,
                n_steps=n_steps,
                method=self.method,
                nfe_per_step=METHODS[self.method]["NFE_PER_STEP"],
                batch_element=i,
                extra_args=extra_args,
                callback=callback,
            )

            result = scipy.integrate.solve_ivp(
                fun=term,
                t_span=[t_max, t_min],
                y0=x[i].reshape(-1),
                method=self.method[3:],
                t_eval=[t_min],
                dense_output=False,
                events=None,
                vectorized=False,
                args=None,
                atol=self.atol,
                rtol=self.rtol,
            )

            if result.success:
                term.progress_bar.update(term.progress_bar.total - term.progress_bar.n)
                term.progress_bar.close()
                sample = torch.from_numpy(result.y[:, -1].reshape(o_shape[1:])).to(o_device, dtype=o_dtype)
            else:
                term.progress_bar.close()
                sample = torch.full(o_shape[1:], np.nan, device=o_device, dtype=o_dtype)
                logger.warning(f"Sample #{i} failed with reason: {result.message}")

            samples.append(sample)

        samples = torch.stack(samples)
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

    @torch.no_grad()
    def __call__(self, model, x: torch.Tensor, sigmas: torch.Tensor, extra_args=None, callback=None, disable=None):
        if self.step_size_controller == "adaptive_pid":
            if self.method not in ADAPTIVE_METHODS:
                raise ValueError("adaptive_pid only supports methods starting with `a`")
            return self._call_torchode(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable)
        elif self.step_size_controller == "fixed_scheduled":
            if (self.method not in ADAPTIVE_METHODS) and (self.method not in FIXED_METHODS):
                raise ValueError("fixed_scheduled only supports methods starting with `a` or `f`")
            return self._call_torchode(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable)
        elif self.step_size_controller == "adaptive_scipy":
            if self.method not in ADAPTIVE_SCIPY_METHODS:
                raise ValueError("adaptive_scipy only supports methods starting with `s`")
            return self._call_scipy(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable)
        else:
            raise NotImplementedError()


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
