import numpy as np
import torch


class SciPyODETerm:
    def __init__(
        self,
        model,
        c_device,
        c_dtype,
        o_device,
        o_dtype,
        o_shape,
        min_sigma,
        t_max,
        t_min,
        n_steps,
        nfe_per_step,
        progress_bar,
        extra_args=None,
        callback=None,
    ):
        self.model = model
        self.c_device = c_device
        self.c_dtype = c_dtype
        self.o_device = o_device
        self.o_dtype = o_dtype
        self.o_shape = o_shape
        self.min_sigma = min_sigma
        self.t_max = t_max
        self.t_min = t_min
        self.n_steps = n_steps
        self.nfe_per_step = nfe_per_step
        self.progress_bar = progress_bar
        self.extra_args = extra_args or {}
        self.callback = callback
        self.step = 0
        self.nfe_step = 0

    def _callback(self, t, y, denoised, mask):
        progress = (self.t_max - t) / (self.t_max - self.t_min)
        percentage = progress * 100
        self.progress_bar.update(percentage - self.step)
        self.step = percentage
        i = round(progress * self.n_steps)

        if self.callback is not None:
            samples = y if mask else denoised
            self.callback(
                {
                    "x": torch.from_numpy(y[np.newaxis, ...]).to(self.o_device, dtype=self.o_dtype),
                    "i": i - 1,
                    "sigma": t,
                    "sigma_hat": t,
                    "denoised": torch.from_numpy(samples[np.newaxis, ...]).to(self.o_device, dtype=self.o_dtype),
                }
            )

    def __call__(self, t, y):
        y = y.reshape(self.o_shape[1:])
        mask = t <= self.min_sigma

        if mask:
            denoised = np.zeros_like(y)
            d = np.zeros_like(y)
        else:
            y_model = torch.from_numpy(y[np.newaxis, ...]).to(self.o_device, dtype=self.o_dtype)
            t_model = torch.tensor([t], device=self.o_device, dtype=self.o_dtype)
            denoised_model = self.model(y_model, t_model, **self.extra_args)
            denoised = denoised_model[0].to(self.c_device).numpy().astype(self.c_dtype)
            d = (y - denoised) / t

        self.nfe_step += 1
        if self.nfe_step % self.nfe_per_step == 0:
            self._callback(t, y, denoised, mask)

        return d.reshape(-1)
