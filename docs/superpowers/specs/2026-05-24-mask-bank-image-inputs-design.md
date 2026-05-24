# Mask Bank IMAGE Inputs Design

## Goal

Update `FreeFuseMaskBankFromImages` so each mask slot accepts a ComfyUI `IMAGE` input instead of a string path. Users should be able to connect `Load Image` nodes directly to `mask_image_00..09`.

## Scope

- Replace `mask_image_00..09` input types on `FreeFuseMaskBankFromImages` from `STRING` to `IMAGE`.
- Remove path/ref loading behavior from this source node.
- Keep existing `FreeFuseMaskTap` string/ref inputs unchanged.
- Keep package exports and node display name unchanged.

## Behavior

- `mask_image_00..09` map to adapter names in `freefuse_data["adapters"]` order, still capped at 10 slots.
- Missing image inputs are skipped.
- Input tensors use ComfyUI image layout: `B,H,W,C` for batched image tensors or `H,W,C` for direct image tensors.
- Only the first batch item is used when a batched tensor is supplied.
- RGB images convert to a 2D mask using luminance from visible channels.
- RGBA images use alpha when `use_alpha=True`; alpha is direct by default (`invert_alpha=False`) and inverted when requested.
- If `use_alpha=False`, RGBA images use visible RGB luminance.
- Masks are resized with nearest neighbor only when both `width` and `height` are positive.
- Masks are binarized with `threshold` after optional resize.
- Output remains `({"masks": masks, "similarity_maps": {}, "metadata": {"adapter_names": adapter_names, "source": "mask_images"}},)`.

## Tests

- Update mask bank tests to pass tensors instead of temp image paths.
- Cover adapter ordering, alpha default/invert behavior, resize with empty slot skipping, local registration, package export coverage, and empty adapter bank.
- Add direct coverage for RGB tensor conversion and batched tensor first-item handling if not already covered by the updated tests.
- Keep focused verification command: `python freefuse_comfyui/tests/test_mask_tap.py`.

## Non-Goals

- Do not support both path strings and IMAGE tensors in this node.
- Do not add a second path-based node.
- Do not change `FreeFuseMaskTap` behavior.
