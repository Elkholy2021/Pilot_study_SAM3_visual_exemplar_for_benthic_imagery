
import numpy as np
def _normalize_masks_to_numpy(masks, image_hw=None):
    """
    Convert various Ultralytics mask formats to a numpy array shaped (N, H, W) of bool.
    Returns None if it cannot interpret the input.

    - Ultralytics Results.masks.data is typically (N,H,W) torch tensor.
    - Sometimes you may get (H,W) for a single mask.
    """
    if masks is None:
        return None

    # If it's a Ultralytics Masks object, prefer .data
    if hasattr(masks, "data") and masks.data is not None:
        try:
            m = masks.data
            if hasattr(m, "cpu"):
                m = m.cpu().numpy()
            else:
                m = np.asarray(m)
        except Exception:
            return None
    else:
        # Already array-like?
        try:
            m = np.asarray(masks)
        except Exception:
            return None

    # Shape handling
    if m.ndim == 2:
        m = m[None, ...]  # (1,H,W)
    if m.ndim != 3:
        return None

    # If float logits/probs, threshold them
    if m.dtype != np.bool_:
        # Many models output 0/1 floats; sometimes logits. A safe default is > 0.5
        m = m > 0.5

    # Optionally, you can sanity-check size vs image
    if image_hw is not None:
        H, W = image_hw
        if m.shape[1] != H or m.shape[2] != W:
            # Keep it best-effort (don’t crash), but warn via return anyway
            # (Could add resizing here if you want.)
            pass

    return m
