# FreeFuse Mask Bank From Images Design

## Goal

Add a ComfyUI node that builds a `FREEFUSE_MASKS` mask bank directly from user-supplied mask images, one image per LoRA slot. This lets users skip FreeFuse Phase 1 auto mask generation when they already have masks.

## Context

`freefuse_comfyui/nodes/mask_tap.py` already contains `FreeFuseMaskTap`, which exposes and optionally replaces masks from an existing mask bank. It is an editor/replacer, not a source node. It also contains reusable image reference parsing, path resolution, resizing, and binarization helpers.

Downstream nodes expect a mask bank dict with masks keyed by adapter name:

```python
{
    "masks": {
        "adapter_name_1": torch.Tensor,
        "adapter_name_2": torch.Tensor,
    },
    "similarity_maps": {},
    "metadata": {
        "adapter_names": ["adapter_name_1", "adapter_name_2"],
        "source": "mask_images",
    },
}
```

## Proposed Node

Add `FreeFuseMaskBankFromImages` in `freefuse_comfyui/nodes/mask_tap.py`.

Display name: `FreeFuse Mask Bank From Images`

Category: `FreeFuse/Utils`

Return type: `FREEFUSE_MASKS`

Return name: `mask_bank`

## Inputs

Required:

- `freefuse_data: FREEFUSE_DATA`

Optional:

- `mask_image_00` through `mask_image_09`: image refs or paths, matching existing MaskTap slot naming.
- `width`: integer, default `0`, min `0`, step `8`.
- `height`: integer, default `0`, min `0`, step `8`.
- `use_alpha`: boolean, default `True`.
- `invert_alpha`: boolean, default `False`.
- `threshold`: float, default `0.5`, range `0.0` to `1.0`.

`width=0` and `height=0` means keep the image's native mask size. If both are nonzero, resize masks to `(height, width)` with nearest-neighbor sampling. If only one dimension is nonzero, do not resize and print a warning, because single-axis resizing would require an unspecified aspect-ratio policy.

## Slot Mapping

Slot order comes from `freefuse_data["adapters"]`.

- Slot `00` maps to the first adapter.
- Slot `01` maps to the second adapter.
- Continue through slot `09`.
- Adapter name comes from `adapter["name"]` when adapter entries are dicts.
- String adapter entries are accepted as adapter names if present.
- Slots beyond adapter count are ignored.
- Blank or missing slots are skipped without crashing.

Metadata `adapter_names` contains the full ordered adapter list, not only adapters with supplied masks, so downstream/debug tools can preserve LoRA order.

## Image Loading

Keep existing `FreeFuseMaskTap` behavior unchanged.

Refactor the existing `_load_mask_from_image_ref(...)` helper only enough to support explicit options. Its current default behavior remains compatible with MaskTap: inverted alpha, threshold `0.5`, binarized output, required target size.

For the new node:

- Use normal alpha semantics by default: `alpha=255 -> mask=1`, `alpha=0 -> mask=0`.
- `invert_alpha=True` flips alpha semantics.
- If `use_alpha=True` and the image has a non-flat alpha channel, alpha drives the mask even when RGB also has variation.
- If alpha is absent, flat, or `use_alpha=False`, grayscale drives the mask.
- If users want grayscale from an image that also has alpha signal, they can set `use_alpha=False`.
- Convert mask values to `float32` in `[0, 1]`.
- Resize with nearest neighbor when explicit target size is supplied.
- Binarize with `mask >= threshold`.

## Output

The node returns:

```python
{
    "masks": {
        adapter_name: mask_tensor,
    },
    "similarity_maps": {},
    "metadata": {
        "adapter_names": ordered_adapter_names,
        "source": "mask_images",
    },
}
```

Mask tensors are 2D `torch.float32` tensors shaped `[H, W]`.

If no adapters exist, return an empty mask bank with `adapter_names=[]` and `source="mask_images"`.

If an image cannot be resolved or loaded, print a warning naming the slot and adapter, then skip that slot.

## Registration

Register the node in:

- `freefuse_comfyui/nodes/mask_tap.py` local `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`.
- `freefuse_comfyui/nodes/__init__.py` imports, combined node mappings, display mappings, and `__all__`.
- `freefuse_comfyui/__init__.py` imports.

## Tests

Extend `freefuse_comfyui/tests/test_mask_tap.py`.

Test cases:

- Build a mask bank from two image slots and two adapters.
- Verify slot order follows `freefuse_data["adapters"]`.
- Verify normal alpha default maps `alpha=255` to mask `1` and `alpha=0` to mask `0`.
- Verify `invert_alpha=True` flips alpha behavior.
- Verify explicit `width` and `height` resize with nearest neighbor.
- Verify empty slots skip without crashing.
- Verify metadata contains ordered `adapter_names` and `source="mask_images"`.
- Verify existing `FreeFuseMaskTap` alpha behavior remains inverted.

Run the existing lightweight mask tap test script or the equivalent pytest target after implementation.

## Non-Goals

- Do not add combined RGBA or batched-image input mode.
- Do not alter existing `FreeFuseMaskTap` public behavior.
- Do not infer target resolution from latent input in this node.
- Do not create preview image outputs for this source node.
