
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Slider
from PIL import Image
import numpy as np
from Evaluation_SAM3.utils.operations import _normalize_masks_to_numpy

FONT_SIZE = 10


def get_single_bbox_interactive(image_path):
    """
    Show image and let user draw ONE bbox (click-drag-release).
    Auto-closes after first selection and returns [x1, y1, x2, y2] as floats.
    """
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img_np)
    ax.set_title("Draw ONE bbox (click-drag-release). Window will close automatically.")
    ax.axis("off")

    bbox = {"val": None}

    def onselect(eclick, erelease):
        x1, y1 = float(eclick.xdata), float(eclick.ydata)
        x2, y2 = float(erelease.xdata), float(erelease.ydata)

        # Normalize so x1<=x2 and y1<=y2
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        bbox["val"] = [x1, y1, x2, y2]

        # Close immediately after one bbox
        plt.close(fig)

    rect_selector = RectangleSelector(
        ax,
        onselect,
        useblit=True,
        button=[1],          # left mouse only
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=False,   # no resizing after draw (keeps it simple)
        drag_from_anywhere=False,
    )

    plt.show()  # blocks until the window closes
    if bbox["val"] is None:
        raise RuntimeError("No bbox was drawn. Close the window only after drawing one bbox.")
    return bbox["val"]


def overlay_bbox(image_path, bbox, title="Selected Visual Prompt BBox"):
    """Quick visual confirmation of the selected bbox."""
    img = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title)

    x1, y1, x2, y2 = bbox
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2)
    ax.add_patch(rect)
    plt.show()


def plot_prompt_bbox(image_path, bbox, title="Selected Visual Prompt BBox"):
    """Show original image with the single user-drawn prompt bbox."""
    img = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title)

    x1, y1, x2, y2 = bbox
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2, edgecolor="red")
    ax.add_patch(rect)

    ax.text(
        x1,
        max(0, y1 - 2),
        "prompt",
        color="white",
        fontsize=FONT_SIZE,
        bbox=dict(facecolor="red", alpha=0.9, pad=1, edgecolor="none"),
    )

    plt.tight_layout()
    plt.show()



def plot_bboxes_with_scores_bar(
    image_path,
    masks,
    bboxes,
    scores,
    thr=0.5,
    valstep=0.1,
    title="Detections (filtered)",
    mask_alpha=0.35,
    plot_bbox=True,
):
    """
    Plot overlay masks + bounding boxes with interactive threshold slider (0 to 1).
    Filter and display items based on current threshold.

    Args:
        masks: (N,H,W) bool/0-1 numpy array or Ultralytics Masks object (best-effort),
               or None to skip mask overlay.
        bboxes: (N,4) xyxy
        scores: (N,)
    """
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    H, W = img_np.shape[:2]

    masks_np = _normalize_masks_to_numpy(masks, image_hw=(H, W)) if masks is not None else None

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.25)

    ax.imshow(img_np)
    ax.axis("off")

    # Create slider
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, "Threshold", 0, 1, valinit=thr, valstep=valstep)

    def draw(current_thr):
        ax.clear()
        ax.imshow(img_np)
        ax.axis("off")
        ax.set_title(f"{title}  (thr={current_thr:.2f})")

        kept = 0
        n = len(bboxes) if bboxes is not None else 0

        # Colormap for masks (one per instance)
        mask_color = (1.0, 0.0, 0.0)  # Red (R, G, B) in [0, 1] range. Change this tuple to modify color.

        for i, (box, score) in enumerate(zip(bboxes, scores)):
            if score < current_thr:
                continue

            # Mask overlay
            if masks_np is not None and i < masks_np.shape[0]:
                m = masks_np[i]
                # Make a colored overlay via RGBA image
                rgba = np.zeros((H, W, 4), dtype=np.float32)
                rgba[..., 0] = mask_color[0]
                rgba[..., 1] = mask_color[1]
                rgba[..., 2] = mask_color[2]
                rgba[..., 3] = m.astype(np.float32) * mask_alpha
                ax.imshow(rgba)
            x1, y1, x2, y2 = [float(v) for v in box]
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2, edgecolor="blue")

            # Bounding box
            if plot_bbox:
                ax.add_patch(rect)

            ax.text(
                x1,
                max(0, y1 - 2),
                f"visual {score:.2f}",
                color="white",
                fontsize=FONT_SIZE,
                bbox=dict(facecolor="blue", alpha=0.9, pad=1, edgecolor="none"),
            )
            kept += 1

        ax.text(
            10,
            20,
            f"kept: {kept}/{n}",
            color="white",
            fontsize=FONT_SIZE,
            bbox=dict(facecolor="black", alpha=0.5, pad=2, edgecolor="none"),
        )

    def update(_val):
        draw(slider.val)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # Initial draw
    draw(thr)
    plt.show()
