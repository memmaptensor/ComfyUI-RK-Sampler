# ComfyUI-RK-Sampler
![example](./workflows/example_comfyui_rk_sampler.jpg)

#### Batched Runge-Kutta Samplers for [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
Supports most practical Explicit Runge-Kutta (ERK) methods.

## Features
- Parallel ODE solvers for fast batch processing
- Explicit and Embedded Explicit Runge-Kutta methods
- PID controller for adaptive step sizing with tunable settings
- Scheduled controller for fixed step sizing (determined by the given $\sigma$ schedule)

## Installation
### ComfyUI-Manager
```
ComfyUI Manager Menu > Install via Git URL > https://github.com/wootwootwootwoot/ComfyUI-RK-Sampler.git
```

### Manual installation
> From `ComfyUI/custom_nodes` and ComfyUI virtual environment
```
git pull https://github.com/wootwootwootwoot/ComfyUI-RK-Sampler.git
pip install torchode
```

## Usage
![workflow](./workflows/workflow_comfyui_rk_sampler.png)
[Basic workflow](./workflows/workflow_comfyui_rk_sampler.json)

> From `Add Node`
```
sampling > custom_sampling > samplers > Runge-Kutta Sampler
```

### Best defaults
```
Fixed step size
method: fe_ralston3
step_size_controller: fixed_scheduled
scheduler: Align Your Steps
steps: 28
cfg: 20
```
```
Adaptive step size
method: ae_bosh3
step_size_controller: adaptive_pid
log_absolute_tolerance: -3.5
log_relative_tolerance: -2.5
pcoeff: 0
icoeff: 1
dcoeff: 0
norm: rms_norm
enable_dt_min: false
enable_dt_max: true
dt_max: 0
safety: 0.9
factor_min: 0.2
factor_max: 10
max_steps: 2147483647
min_sigma: 0.00001
cfg: 20
```

### Choose a step size controller
- These methods support normal as well as high CFG scales, but the results may depend on the specific model.
- If you don't know the right step count or CFG scale:
  1. Try the `adaptive_pid` controller with the base CFG and increment it until the results get worse.
  2. Tune the CFG to the desired output.
  3. Try the `fixed_scheduled` controller with AYS scheduler at 28 steps using the same CFG.
  4. Tune the scheduler step count from 28 steps.
- For SDXL, the best results I got were from CFG scales between 7-35

### Choose a method
#### Step size controller support
- Methods starting with `a` support both adaptive and fixed step size controllers.
- Methods starting with `f` only support fixed step size controllers.

#### Recommended methods
- Try `ae_bosh3`, `ae_dopri5`, and `ae_fehlberg5` with the PID adaptive step size controller.
- Try `fe_ralston3`, `ae_bosh3`, and `fe_ssprk3` with the fixed step size controller.

#### Quality ranking
Tested on `RTX3090`, `SDXL`, `AYS 28 steps`, `batch size 1`, `896x1152`, `CFG=30`, `fixed_scheduled`
| Rank | Name | Method | Order | NFEs | Time |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| 1 | `fe_ralston3` | Ralston | 3 | 3 | 23.08s |
| 2 | `ae_bosh3` | Bogacki–Shampine | 3 | 3 | 23.03s |
| 3 | `fe_ssprk3` | Strong Stability Preserving Runge-Kutta | 3 | 3 | 22.97s |
| 4 | `fe_kutta4` | Runge-Kutta | 4 | 4 | 29.87s |
| 5 | `fe_kutta_38th4` | Runge-Kutta (3/8-rule) | 4 | 4 | 30.10s |
| 6 | `ae_dopri5` | Dormand–Prince | 5 | 6 | 44.03s |
| 7 | `ae_fehlberg5` | Runge–Kutta–Fehlberg | 5 | 6 | 44.28s |
| 8 | `ae_heun_euler2` | Heun–Euler | 2 | 2 | 15.87s |
| 9 | `fe_kutta3` | Runge-Kutta | 3 | 3 | 22.83s |
| 10 | `ae_ralston2` | Ralston | 2 | 2 | 16.05s |
| 11 | `ae_cash_karp5` | Cash-Karp | 5 | 6 | 44.47s |
| 12 | `fe_ralston4` | Ralston | 4 | 4 | 29.69s |
| 13 | `fe_euler1` | Forward Euler | 1 | 1 | 9.00s |
| 14 | `fe_wray3` | van der Houwen and Wray | 3 | 3 | 22.96s |
| 15 | `fe_heun3` | Heun | 3 | 3 | 23.13s |
| 16 | `ae_tsit5` | Tsitouras | 5 | 6 | 43.90s |
| 17 | `ae_midpoint2` | Implicit midpoint | 2 | 2 | 15.97s |
| 18 | `ae_fehlberg2` | Runge–Kutta–Fehlberg | 2 | 3 | 23.23s |
| 19 | `ae_dopri8` | Dormand–Prince | 8 | 13 | 94.10s |

### Solver settings
| Option | Description |
| ----------- | ----------- |
| method | Solver method. |
| step_size_controller | Step size controller. **The rest of the settings only apply when `adaptive_pid` is selected.** For `fixed_scheduled`, the step count is determined by your scheduler. |
| log_absolute_tolerance | $log_{10}$ of the threshold below which you do not worry about the accuracy of the solution since it is effectively 0. More negative $log_{10}$ values correspond to tighter tolerances and higher quality results. |
| log_relative_tolerance | $log_{10}$ of the threshold for the relative error of a single step of the integrator. `log_relative_tolerance` cannot be more negative than `log_absolute_tolerance`. In practice, set the value for `log_relative_tolerance` to be 1 higher than `log_absolute_tolerance`. |
| pcoeff | Coefficients for the proportional term of the PID controller. | 
| icoeff | Coefficients for the integral term of the PID controller. P/I/D of 0/1/0 corresponds to a basic integral controller. | 
| dcoeff | Coefficients for the derivative term of the PID controller. | 
| norm | Normalization function for error control. Step sizes are chosen so that `norm(error / (absolute_tolerance + relative_tolerance * y))` is approximately one. |
| enable_dt_min | Enable clamping of the minimum step size to take to `dt_min`. |
| enable_dt_max | Enable clamping of the maximum step size to take to `dt_max`. |
| dt_min | The `dt_max` value to clamp to. Since we are solving a reverse-time ODE, this value should be negative. |
| dt_max | The `dt_min` value to clamp to. Since we are solving a reverse-time ODE, this value should be negative. Clamped to 0 by default to force a monotonic solve. |
| safety | Multiplicative safety factor. |
| factormin | Minimum amount a step size can be decreased relative to the previous step. |
| factormax | Maximum amount a step size can be increased relative to the previous step. |
| max_steps | Maximum amount of steps an adaptive step size controller is allowed to take. Taking more steps than `max_steps` will return an error. **Does not apply to fixed step size controllers.** |
| min_sigma | Lower bound for $\sigma$ to consider the IVP solve to be complete. |

## Comparison
### Related projects
This extension improves upon [ComfyUI-ODE](https://github.com/redhottensors/ComfyUI-ODE) by adding support for parallel processing, more controllability, high-quality live previews, a PID controller, and support for more fixed and adaptive step size solvers.

### Speed 
Tested on `RTX3090`, `SDXL`, `896x1152`, `CFG=30`, `ae_bosh3 (ComfyUI-RK-Sampler) / bosh3 (ComfyUI-ODE)`, `adaptive_pid 0/1/0`, `log_absolute_tolerance=-3.5`, `log_relative_tolerance=-2.5`
| Batch size | ComfyUI-RK-Sampler | ComfyUI-ODE |
| ----------- | ----------- | ----------- |
| 1 | 1m2s | 1m6s |
| 2 | 2m3s | 2m7s |
| 4 | 4m4s | 4m27s |
| 8 | 7m39s | 9m3s |
| 16 | 15m29s | 17m22s |
| 32 | 30m18s | 36m34s |
 
## Explanation
> From [Wikipedia](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods):
>
> In numerical analysis, the Runge–Kutta methods are a family of implicit and explicit iterative methods, which include the Euler method, used in temporal discretization for the approximate solutions of simultaneous nonlinear equations. These methods were developed around 1900 by the German mathematicians Carl Runge and Wilhelm Kutta.

The Runge-Kutta methods are a family of methods used for solving approximate solutions of ODEs by iterative discretization (or, if in diffusion terms, by sampling).

Runge-Kutta methods generally have less discretization error than standard diffusion sampling methods, allowing for the use of high CFG scales (within practical limits) to create high-quality results without artifacts.