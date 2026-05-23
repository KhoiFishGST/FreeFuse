#!/usr/bin/env python
"""Lightweight routing tests for FreeFuse attention bias patches."""

import importlib.util
import os
import sys
import types
import unittest

import torch


def _install_comfy_stubs():
    comfy = types.ModuleType("comfy")
    comfy_ops = types.ModuleType("comfy.ops")
    comfy_ops.scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention

    comfy_ldm = types.ModuleType("comfy.ldm")
    comfy_ldm_flux = types.ModuleType("comfy.ldm.flux")
    comfy_ldm_flux_math = types.ModuleType("comfy.ldm.flux.math")
    comfy_ldm_flux_math.apply_rope = lambda q, k, pe: (q, k)

    comfy_ldm_lumina = types.ModuleType("comfy.ldm.lumina")
    comfy_ldm_lumina_model = types.ModuleType("comfy.ldm.lumina.model")
    comfy_ldm_lumina_model.modulate = lambda x, scale, timestep_zero_index=None: x
    comfy_ldm_lumina_model.apply_gate = lambda gate, x, timestep_zero_index=None: x * gate
    comfy_ldm_lumina_model.clamp_fp16 = lambda x: x

    comfy.ops = comfy_ops

    sys.modules.update({
        "comfy": comfy,
        "comfy.ops": comfy_ops,
        "comfy.ldm": comfy_ldm,
        "comfy.ldm.flux": comfy_ldm_flux,
        "comfy.ldm.flux.math": comfy_ldm_flux_math,
        "comfy.ldm.lumina": comfy_ldm_lumina,
        "comfy.ldm.lumina.model": comfy_ldm_lumina_model,
    })


def _load_attention_bias_patch_module():
    _install_comfy_stubs()
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    freefuse_comfyui_dir = os.path.dirname(tests_dir)
    module_path = os.path.join(
        freefuse_comfyui_dir, "freefuse_core", "attention_bias_patch.py"
    )

    package = types.ModuleType("freefuse_comfyui")
    package.__path__ = [freefuse_comfyui_dir]
    core_package = types.ModuleType("freefuse_comfyui.freefuse_core")
    core_package.__path__ = [os.path.join(freefuse_comfyui_dir, "freefuse_core")]
    sys.modules.setdefault("freefuse_comfyui", package)
    sys.modules.setdefault("freefuse_comfyui.freefuse_core", core_package)

    spec = importlib.util.spec_from_file_location(
        "freefuse_comfyui.freefuse_core.attention_bias_patch",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyDiffusionModel:
    def __init__(self, n_single_transformer=0, n_layers=0):
        self.single_transformer_blocks = [object() for _ in range(n_single_transformer)]
        self.layers = [object() for _ in range(n_layers)]


class DummyInnerModel:
    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model


class DummyModelPatcher:
    def __init__(self, diffusion_model):
        self.model = DummyInnerModel(diffusion_model)
        self.model_options = {"transformer_options": {}}
        self.patch_replace_calls = []

    def set_model_patch_replace(self, fn, model_part, block_kind, block_index):
        self.patch_replace_calls.append((model_part, block_kind, int(block_index), fn))


class TestAttentionBiasRouting(unittest.TestCase):
    def test_flux2_single_transformer_blocks_receive_bias_patches(self):
        module = _load_attention_bias_patch_module()
        model = DummyModelPatcher(DummyDiffusionModel(n_single_transformer=3))
        config = module.AttentionBiasConfig(enabled=True, apply_to_blocks=None)

        module.apply_attention_bias_patches(
            model_patcher=model,
            attention_bias=None,
            config=config,
            txt_seq_len=8,
            model_type="flux2",
            lora_masks={"a": torch.ones(1, 4)},
            token_pos_maps={"a": [[0, 1]]},
        )

        patched = [(kind, idx) for _, kind, idx, _ in model.patch_replace_calls]
        self.assertEqual(patched, [("single_block", 0), ("single_block", 1), ("single_block", 2)])

    def test_flux2_default_double_stream_preset_falls_back_to_single_blocks(self):
        module = _load_attention_bias_patch_module()
        model = DummyModelPatcher(DummyDiffusionModel(n_single_transformer=2))
        config = module.AttentionBiasConfig(enabled=True, apply_to_blocks="double_stream_only")

        module.apply_attention_bias_patches(
            model_patcher=model,
            attention_bias=None,
            config=config,
            txt_seq_len=8,
            model_type="flux2",
            lora_masks={"a": torch.ones(1, 4)},
            token_pos_maps={"a": [[0, 1]]},
        )

        patched = [(kind, idx) for _, kind, idx, _ in model.patch_replace_calls]
        self.assertEqual(patched, [("single_block", 0), ("single_block", 1)])

    def test_z_image_registers_optimized_attention_override(self):
        module = _load_attention_bias_patch_module()
        model = DummyModelPatcher(DummyDiffusionModel(n_layers=3))
        config = module.AttentionBiasConfig(enabled=True, apply_to_blocks=None)

        module.apply_attention_bias_patches(
            model_patcher=model,
            attention_bias=None,
            config=config,
            txt_seq_len=8,
            model_type="z_image",
            lora_masks={"a": torch.ones(1, 4)},
            token_pos_maps={"a": [[0, 1]]},
        )

        override = model.model_options["transformer_options"].get("optimized_attention_override")
        self.assertTrue(callable(override))

    def test_z_image_override_skips_calls_without_main_layer_context(self):
        module = _load_attention_bias_patch_module()
        model = DummyModelPatcher(DummyDiffusionModel(n_layers=3))
        config = module.AttentionBiasConfig(enabled=True, apply_to_blocks=None)
        module.apply_attention_bias_patches(
            model_patcher=model,
            attention_bias=None,
            config=config,
            txt_seq_len=8,
            model_type="z_image",
            lora_masks={"a": torch.ones(1, 4)},
            token_pos_maps={"a": [[0, 1]]},
        )
        override = model.model_options["transformer_options"]["optimized_attention_override"]
        seen = {}

        def original_fn(q, k, v, heads, mask=None, *args, transformer_options=None, **kwargs):
            seen["mask"] = mask
            return q

        q = torch.zeros(1, 1, 10, 2)
        override(original_fn, q, q, q, 1, None, transformer_options={})

        self.assertIsNone(seen["mask"])


if __name__ == "__main__":
    unittest.main()
