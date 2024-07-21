"""
@author: wootwootwootwoot
@title: ComfyUI-RK-Sampler
@nickname: ComfyUI-RK-Sampler
@description: Batched Runge-Kutta Samplers for ComfyUI
"""

from .nodes import nodes_rk_sampler

NODE_CLASS_MAPPINGS = {
    **nodes_rk_sampler.NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **nodes_rk_sampler.NODE_DISPLAY_NAME_MAPPINGS,
}
