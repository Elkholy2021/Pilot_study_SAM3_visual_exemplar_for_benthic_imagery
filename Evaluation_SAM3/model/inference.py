from ultralytics.models.sam import SAM3SemanticPredictor
import numpy as np

def load_SAM3_predictor(  conf=0.25,
                        task="segment",
                        mode="predict",
                        model="/media/hassan/Data_vault1/datasets_hassan/models/sam3.pt",
                        half=True,
                        save=True
                        ):
    overrides = dict(
        conf=conf,
        task=task,
        mode=mode,
        model=model,
        half=half,
        save=save,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    return predictor
def run_SAM3_visual_exempler_prompt(results):
    """
    Run helper to extract masks, boxes, scores from Ultralytics Results object.
    """
    try:
        # results is often a list-like of Results
        r0 = results[0] if isinstance(results, (list, tuple)) else results

        # Boxes
        if not hasattr(r0, "boxes") or r0.boxes is None:
            return None, None, None

        boxes = r0.boxes
        if not hasattr(boxes, "xyxy") or not hasattr(boxes, "conf"):
            return None, None, None

        bboxes = boxes.xyxy
        scores = boxes.conf

        # Convert to numpy
        if hasattr(bboxes, "cpu"):
            bboxes = bboxes.cpu().numpy()
        else:
            bboxes = np.asarray(bboxes)

        if hasattr(scores, "cpu"):
            scores = scores.cpu().numpy()
        else:
            scores = np.asarray(scores)

        # Masks (optional)
        masks = None
        if hasattr(r0, "masks") and r0.masks is not None:
            # Keep as Ultralytics masks object here; downstream normalizes robustly
            masks = r0.masks

        return masks, bboxes, scores
    except Exception:
        return None, None, None
    



def run_SAM3_predictor(predictor,image_path, bbox):


    predictor.set_image(image_path)
    results = predictor(bboxes=[bbox])
    return results