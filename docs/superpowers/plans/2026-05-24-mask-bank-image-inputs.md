# Mask Bank IMAGE Inputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change `FreeFuseMaskBankFromImages` so users connect ComfyUI `IMAGE` outputs directly to `mask_image_00..09` instead of entering image paths.

**Architecture:** Keep `FreeFuseMaskTap` path/ref behavior unchanged. Add a tensor-to-mask helper beside `_load_mask_from_image_ref`, then update only `FreeFuseMaskBankFromImages` to accept `IMAGE` slots and call the tensor helper. Preserve adapter ordering, resizing, thresholding, metadata, and package exports.

**Tech Stack:** Python, PyTorch tensors, ComfyUI `IMAGE` layout (`B,H,W,C` or `H,W,C`), existing lightweight test script `freefuse_comfyui/tests/test_mask_tap.py`.

---

## File Structure

- Modify `freefuse_comfyui/tests/test_mask_tap.py`: replace path-based source-node tests with ComfyUI tensor input tests and add input type coverage.
- Modify `freefuse_comfyui/nodes/mask_tap.py`: add `_load_mask_from_image_tensor`, change `FreeFuseMaskBankFromImages.INPUT_TYPES()` to `IMAGE`, and update `build_mask_bank()` to consume tensors instead of paths.
- Verify `freefuse_comfyui/nodes/__init__.py` and `freefuse_comfyui/__init__.py`: no code changes expected because exports and display name stay the same.

## Task 1: Add Failing IMAGE Input Tests

**Files:**
- Modify: `freefuse_comfyui/tests/test_mask_tap.py:41-430`
- Test: `freefuse_comfyui/tests/test_mask_tap.py`

- [ ] **Step 1: Add tensor image helpers after `_save_split_alpha`**

Insert this code after `_save_split_alpha`:

```python
def _solid_rgb_image(height, width, value, *, batch=True):
    img = torch.full((height, width, 3), float(value), dtype=torch.float32)
    return img.unsqueeze(0) if batch else img


def _checker_rgb_image(*, batch=True):
    img = torch.tensor(
        [
            [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    return img.unsqueeze(0) if batch else img


def _split_rgba_image(left_alpha, right_alpha):
    img = torch.zeros((1, 2, 4, 4), dtype=torch.float32)
    img[:, :, :2, :3] = 1.0
    img[:, :, 2:, :3] = 0.0
    img[:, :, :2, 3] = float(left_alpha)
    img[:, :, 2:, 3] = float(right_alpha)
    return img
```

- [ ] **Step 2: Replace the adapter-order test with tensor inputs**

Replace `test_mask_bank_from_images_builds_bank_in_adapter_order` with:

```python
def test_mask_bank_from_images_builds_bank_in_adapter_order():
    mod = _module()
    node = mod.FreeFuseMaskBankFromImages()

    out, = node.build_mask_bank(
        freefuse_data=_freefuse_data("first", "second"),
        mask_image_00=_solid_rgb_image(2, 3, 1.0),
        mask_image_01=_solid_rgb_image(2, 3, 0.0),
    )

    assert list(out["masks"].keys()) == ["first", "second"]
    assert torch.allclose(out["masks"]["first"], torch.ones(2, 3), atol=1e-6)
    assert torch.allclose(out["masks"]["second"], torch.zeros(2, 3), atol=1e-6)
    assert out["similarity_maps"] == {}
    assert out["metadata"]["adapter_names"] == ["first", "second"]
    assert out["metadata"]["source"] == "mask_images"
```

- [ ] **Step 3: Replace the alpha test with RGBA tensor coverage**

Replace `test_mask_bank_from_images_alpha_defaults_and_invert` with:

```python
def test_mask_bank_from_images_alpha_defaults_invert_and_rgb_fallback():
    mod = _module()
    node = mod.FreeFuseMaskBankFromImages()
    image = _split_rgba_image(left_alpha=0.0, right_alpha=1.0)

    normal, = node.build_mask_bank(
        freefuse_data=_freefuse_data("alpha_lora"),
        mask_image_00=image,
    )
    inverted, = node.build_mask_bank(
        freefuse_data=_freefuse_data("alpha_lora"),
        mask_image_00=image,
        invert_alpha=True,
    )
    rgb_fallback, = node.build_mask_bank(
        freefuse_data=_freefuse_data("alpha_lora"),
        mask_image_00=image,
        use_alpha=False,
    )

    normal_mask = normal["masks"]["alpha_lora"]
    inverted_mask = inverted["masks"]["alpha_lora"]
    rgb_mask = rgb_fallback["masks"]["alpha_lora"]

    assert torch.allclose(normal_mask[:, :2], torch.zeros(2, 2), atol=1e-6)
    assert torch.allclose(normal_mask[:, 2:], torch.ones(2, 2), atol=1e-6)
    assert torch.allclose(inverted_mask[:, :2], torch.ones(2, 2), atol=1e-6)
    assert torch.allclose(inverted_mask[:, 2:], torch.zeros(2, 2), atol=1e-6)
    assert torch.allclose(rgb_mask[:, :2], torch.ones(2, 2), atol=1e-6)
    assert torch.allclose(rgb_mask[:, 2:], torch.zeros(2, 2), atol=1e-6)
```

- [ ] **Step 4: Replace the resize/skip test with a direct `H,W,C` tensor**

Replace `test_mask_bank_from_images_resizes_and_skips_empty_slots` with:

```python
def test_mask_bank_from_images_resizes_and_skips_empty_slots():
    mod = _module()
    node = mod.FreeFuseMaskBankFromImages()

    out, = node.build_mask_bank(
        freefuse_data=_freefuse_data("empty_slot", "filled_slot"),
        mask_image_01=_checker_rgb_image(batch=False),
        width=4,
        height=4,
    )

    assert "empty_slot" not in out["masks"]
    assert list(out["masks"].keys()) == ["filled_slot"]
    mask = out["masks"]["filled_slot"]
    assert mask.shape == (4, 4)
    assert torch.allclose(mask[:2, :2], torch.ones(2, 2), atol=1e-6)
    assert torch.allclose(mask[:2, 2:], torch.zeros(2, 2), atol=1e-6)
    assert torch.allclose(mask[2:, :2], torch.zeros(2, 2), atol=1e-6)
    assert torch.allclose(mask[2:, 2:], torch.ones(2, 2), atol=1e-6)
    assert out["metadata"]["adapter_names"] == ["empty_slot", "filled_slot"]
```

- [ ] **Step 5: Add first-batch-item test before registration tests**

Insert this test before `test_mask_bank_from_images_local_registration`:

```python
def test_mask_bank_from_images_uses_first_image_from_batch():
    mod = _module()
    node = mod.FreeFuseMaskBankFromImages()
    batch = torch.stack(
        [
            torch.zeros((2, 2, 3), dtype=torch.float32),
            torch.ones((2, 2, 3), dtype=torch.float32),
        ],
        dim=0,
    )

    out, = node.build_mask_bank(
        freefuse_data=_freefuse_data("batched"),
        mask_image_00=batch,
    )

    assert torch.allclose(out["masks"]["batched"], torch.zeros(2, 2), atol=1e-6)
```

- [ ] **Step 6: Add IMAGE input type assertions to local registration test**

Replace `test_mask_bank_from_images_local_registration` with:

```python
def test_mask_bank_from_images_local_registration():
    mod = _module()

    assert mod.NODE_CLASS_MAPPINGS["FreeFuseMaskBankFromImages"] is mod.FreeFuseMaskBankFromImages
    assert mod.NODE_DISPLAY_NAME_MAPPINGS["FreeFuseMaskBankFromImages"] == "FreeFuse Mask Bank From Images"

    inputs = mod.FreeFuseMaskBankFromImages.INPUT_TYPES()
    for i in range(10):
        assert inputs["optional"][f"mask_image_{i:02d}"] == ("IMAGE",)
```

- [ ] **Step 7: Update `run_all_tests()` test name**

Replace the alpha test call and add the first-batch test call:

```python
    test_mask_bank_from_images_alpha_defaults_invert_and_rgb_fallback()
    test_mask_bank_from_images_resizes_and_skips_empty_slots()
    test_mask_bank_from_images_uses_first_image_from_batch()
    test_mask_bank_from_images_local_registration()
```

- [ ] **Step 8: Run tests to verify RED**

Run: `python freefuse_comfyui/tests/test_mask_tap.py`

Expected: FAIL before implementation. Acceptable first failure is either:

```text
AssertionError
```

from empty masks because tensor slots are ignored, or an assertion showing `mask_image_00` is still `("STRING", {"default": "", "multiline": False})` instead of `("IMAGE",)`.

- [ ] **Step 9: Commit failing tests**

```bash
git add freefuse_comfyui/tests/test_mask_tap.py
git commit -m "test: expect mask bank image tensors"
```

## Task 2: Switch Source Node To IMAGE Tensors

**Files:**
- Modify: `freefuse_comfyui/nodes/mask_tap.py:227-429`
- Test: `freefuse_comfyui/tests/test_mask_tap.py`

- [ ] **Step 1: Add `_load_mask_from_image_tensor` after `_load_mask_from_image_ref`**

Insert this helper after the `_load_mask_from_image_ref` function:

```python
def _load_mask_from_image_tensor(
    image_tensor,
    target_h: Optional[int] = None,
    target_w: Optional[int] = None,
    *,
    use_alpha: bool = True,
    invert_alpha: bool = False,
    threshold: float = _MASK_BINARY_THRESHOLD,
    binarize: bool = True,
    warning_prefix: str = "[FreeFuseMaskBankFromImages]",
) -> Optional[torch.Tensor]:
    if not isinstance(image_tensor, torch.Tensor):
        return None

    try:
        img = image_tensor.detach().float()
        if img.dim() == 4:
            if img.shape[0] < 1:
                return None
            img = img[0]
        if img.dim() != 3 or img.shape[-1] < 1:
            return None

        channels = int(img.shape[-1])
        if bool(use_alpha) and channels >= 4:
            alpha = img[..., 3].clamp(0.0, 1.0)
            mask_2d = 1.0 - alpha if bool(invert_alpha) else alpha
        elif channels >= 3:
            rgb = img[..., :3].clamp(0.0, 1.0)
            mask_2d = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
        else:
            mask_2d = img[..., 0].clamp(0.0, 1.0)

        mask_2d = mask_2d.float()
        if target_h is not None and target_w is not None:
            mask_2d = _resize_2d(mask_2d, int(target_h), int(target_w), mode="nearest")
        if bool(binarize):
            mask_2d = _binarize_2d(mask_2d, threshold)
        return mask_2d
    except Exception as e:
        print(f"{warning_prefix} Warning: failed to read image tensor: {e}")
        return None
```

- [ ] **Step 2: Change source node input slots from `STRING` to `IMAGE`**

In `FreeFuseMaskBankFromImages.INPUT_TYPES()`, replace the ten mask image optional entries with:

```python
                "mask_image_00": ("IMAGE",),
                "mask_image_01": ("IMAGE",),
                "mask_image_02": ("IMAGE",),
                "mask_image_03": ("IMAGE",),
                "mask_image_04": ("IMAGE",),
                "mask_image_05": ("IMAGE",),
                "mask_image_06": ("IMAGE",),
                "mask_image_07": ("IMAGE",),
                "mask_image_08": ("IMAGE",),
                "mask_image_09": ("IMAGE",),
```

- [ ] **Step 3: Replace source node build loop path parsing with tensor loading**

Inside `FreeFuseMaskBankFromImages.build_mask_bank()`, replace the loop body from `image_ref = kwargs.get(key)` through `masks[adapter_name] = mask.float()` with:

```python
            image_tensor = kwargs.get(key)
            if image_tensor is None:
                continue

            mask = _load_mask_from_image_tensor(
                image_tensor=image_tensor,
                target_h=target_h,
                target_w=target_w,
                use_alpha=bool(use_alpha),
                invert_alpha=bool(invert_alpha),
                threshold=float(threshold),
                binarize=True,
                warning_prefix="[FreeFuseMaskBankFromImages]",
            )
            if mask is None:
                print(f"[FreeFuseMaskBankFromImages] Warning: failed to read {key} for adapter '{adapter_name}'")
                continue
            masks[adapter_name] = mask.float()
```

- [ ] **Step 4: Confirm `FreeFuseMaskTap` still uses path/ref inputs**

Inspect `FreeFuseMaskTap.INPUT_TYPES()` and confirm its `mask_image_00..09` entries remain:

```python
                "mask_image_00": ("STRING", {"default": "", "multiline": False}),
```

The exact same `STRING` shape should remain for all ten `FreeFuseMaskTap` slots.

- [ ] **Step 5: Run tests to verify GREEN**

Run: `python freefuse_comfyui/tests/test_mask_tap.py`

Expected:

```text
All mask tap/reassemble tests passed.
```

- [ ] **Step 6: Commit implementation**

```bash
git add freefuse_comfyui/nodes/mask_tap.py freefuse_comfyui/tests/test_mask_tap.py
git commit -m "feat: accept image tensors for mask banks"
```

## Task 3: Final Verification And PR Update

**Files:**
- Verify: `freefuse_comfyui/nodes/mask_tap.py`
- Verify: `freefuse_comfyui/tests/test_mask_tap.py`
- Verify: `freefuse_comfyui/nodes/__init__.py`
- Verify: `freefuse_comfyui/__init__.py`

- [ ] **Step 1: Run focused test script**

Run: `python freefuse_comfyui/tests/test_mask_tap.py`

Expected:

```text
All mask tap/reassemble tests passed.
```

- [ ] **Step 2: Compile changed Python files**

Run: `python -m py_compile freefuse_comfyui/nodes/mask_tap.py freefuse_comfyui/tests/test_mask_tap.py`

Expected: exit code 0 and no output.

- [ ] **Step 3: Check diff whitespace**

Run: `git diff --check master...HEAD`

Expected: exit code 0 and no output.

- [ ] **Step 4: Check worktree status**

Run: `git status --short`

Expected: no staged or unstaged tracked changes. Pre-existing untracked `docs/superpowers/plans/2026-05-23-mask-bank-from-images.md` may remain; do not commit or remove it unless the human explicitly asks.

- [ ] **Step 5: Review recent commits**

Run: `git log --oneline -10`

Expected: latest commits include:

```text
feat: accept image tensors for mask banks
test: expect mask bank image tensors
docs: design mask image tensor inputs
feat: export mask image bank node
feat: build mask banks from images
```

- [ ] **Step 6: Push PR branch**

Run: `git push`

Expected: branch `feature/mask-bank-from-images` pushes to `origin/feature/mask-bank-from-images`, updating PR `https://github.com/KhoiFishGST/FreeFuse/pull/2`.
