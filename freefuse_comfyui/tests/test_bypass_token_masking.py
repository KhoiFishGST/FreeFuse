#!/usr/bin/env python
"""Focused tests for FreeFuse bypass token masking.

This test stubs the small ComfyUI surface needed to load
``bypass_lora_loader.py`` so it can run without a full ComfyUI checkout.
"""

import importlib.util
import os
import sys
import types
import unittest

import torch
import torch.nn as nn


def _install_comfy_stubs():
    comfy = types.ModuleType("comfy")

    lora = types.ModuleType("comfy.lora")
    lora_convert = types.ModuleType("comfy.lora_convert")
    model_management = types.ModuleType("comfy.model_management")

    weight_adapter = types.ModuleType("comfy.weight_adapter")
    weight_adapter_base = types.ModuleType("comfy.weight_adapter.base")
    weight_adapter_bypass = types.ModuleType("comfy.weight_adapter.bypass")

    class WeightAdapterBase:
        def bypass_forward(self, original_forward, x, *args, **kwargs):
            return original_forward(x, *args, **kwargs)

    class WeightAdapterTrainBase(WeightAdapterBase):
        pass

    class BypassForwardHook:
        def __init__(self, module, adapter, multiplier=1.0):
            self.module = module
            self.adapter = adapter
            self.multiplier = multiplier
            self.original_forward = None

    def get_module_type_info(module):
        return {
            "is_conv": isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)),
            "conv_dim": 0,
            "kernel_size": getattr(module, "kernel_size", (1,)),
            "in_channels": getattr(module, "in_channels", None),
            "out_channels": getattr(module, "out_channels", None),
            "stride": getattr(module, "stride", (1,)),
            "padding": getattr(module, "padding", (0,)),
            "dilation": getattr(module, "dilation", (1,)),
            "groups": getattr(module, "groups", 1),
        }

    weight_adapter_base.WeightAdapterBase = WeightAdapterBase
    weight_adapter_base.WeightAdapterTrainBase = WeightAdapterTrainBase
    weight_adapter.WeightAdapterBase = WeightAdapterBase
    weight_adapter.WeightAdapterTrainBase = WeightAdapterTrainBase
    weight_adapter.bypass = weight_adapter_bypass

    weight_adapter_bypass.BypassForwardHook = BypassForwardHook
    weight_adapter_bypass.get_module_type_info = get_module_type_info

    patcher_extension = types.ModuleType("comfy.patcher_extension")

    class PatcherInjection:
        def __init__(self, inject=None, eject=None):
            self.inject = inject
            self.eject = eject

    patcher_extension.PatcherInjection = PatcherInjection

    comfy.lora = lora
    comfy.lora_convert = lora_convert
    comfy.model_management = model_management
    comfy.weight_adapter = weight_adapter

    sys.modules.update({
        "comfy": comfy,
        "comfy.lora": lora,
        "comfy.lora_convert": lora_convert,
        "comfy.model_management": model_management,
        "comfy.weight_adapter": weight_adapter,
        "comfy.weight_adapter.base": weight_adapter_base,
        "comfy.weight_adapter.bypass": weight_adapter_bypass,
        "comfy.patcher_extension": patcher_extension,
    })


def _load_bypass_module():
    _install_comfy_stubs()
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    freefuse_comfyui_dir = os.path.dirname(tests_dir)
    module_path = os.path.join(
        freefuse_comfyui_dir, "freefuse_core", "bypass_lora_loader.py"
    )
    spec = importlib.util.spec_from_file_location("bypass_lora_loader_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ZeroBaseModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4, 4))

    def forward(self, x):
        return torch.zeros_like(x)


class OnesAdapter:
    def h(self, x, base_out):
        return torch.ones_like(base_out)

    def g(self, value):
        return value


class TestBypassTokenMasking(unittest.TestCase):
    def test_token_masking_zeroes_other_adapter_text_positions(self):
        module = ZeroBaseModule()
        bypass = _load_bypass_module()
        hook = bypass.MultiAdapterBypassForwardHook(
            module,
            module_key="diffusion_model.single_transformer_blocks.0.attn.to_q",
        )
        hook.add_adapter(OnesAdapter(), adapter_name="adapter_a")
        hook.set_masks({"adapter_a": torch.ones(2, 2)}, latent_size=(2, 2), txt_len=2)
        hook.set_token_pos_maps({"adapter_a": [[0]], "adapter_b": [[1]]})
        hook.inject()

        output = module(torch.zeros(1, 6, 4))

        self.assertTrue(torch.allclose(output[0, 0], torch.ones(4)))
        self.assertTrue(torch.allclose(output[0, 1], torch.zeros(4)))
        self.assertTrue(torch.allclose(output[0, 2:], torch.ones(4, 4)))

    def test_manager_stores_token_positions_for_deferred_injection(self):
        bypass = _load_bypass_module()
        manager = bypass.OffsetBypassInjectionManager()
        token_pos_maps = {"adapter_a": [[0]], "adapter_b": [[1]]}

        manager.set_masks(
            {"adapter_a": torch.ones(2, 2)},
            latent_size=(2, 2),
            txt_len=2,
            token_pos_maps=token_pos_maps,
        )

        self.assertEqual(manager._pending_token_pos_maps, token_pos_maps)


if __name__ == "__main__":
    unittest.main()
