import numpy as np
from scipy.integrate._ivp.rk import rk_step

DT_MIN = None
DT_MAX = None
SAFETY = None
MIN_FACTOR = None
MAX_FACTOR = None
MAX_STEPS = None
N_STEPS = None
TERM = None


def _step_impl(self):
    t = self.t
    y = self.y

    rtol = self.rtol
    atol = self.atol

    if DT_MIN is not None or DT_MAX is not None:
        h_abs = np.abs(np.clip(self.direction * self.h_abs, DT_MIN, DT_MAX))

    step_accepted = False
    step_rejected = False

    while not step_accepted:
        global N_STEPS
        if N_STEPS >= MAX_STEPS:
            return False, "Reached max_steps"
        N_STEPS += 1

        h = h_abs * self.direction
        t_new = t + h

        if self.direction * (t_new - self.t_bound) > 0:
            t_new = self.t_bound

        h = t_new - t
        h_abs = np.abs(h)

        y_new, f_new = rk_step(self.fun, t, y, self.f, h, self.A, self.B, self.C, self.K)
        scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
        error_norm = self._estimate_error_norm(self.K, h, scale)

        if error_norm < 1:
            if error_norm == 0:
                factor = MAX_FACTOR
            else:
                factor = min(MAX_FACTOR, SAFETY * error_norm**self.error_exponent)

            if step_rejected:
                factor = min(1, factor)

            h_abs *= factor

            step_accepted = True
        else:
            h_abs *= max(MIN_FACTOR, SAFETY * error_norm**self.error_exponent)
            step_rejected = True

        TERM.trigger_callback(t_new, y_new)

    self.h_previous = h
    self.y_old = y

    self.t = t_new
    self.y = y_new

    self.h_abs = h_abs
    self.f = f_new

    return True, None
