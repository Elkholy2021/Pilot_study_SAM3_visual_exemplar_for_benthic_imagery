from Evaluation_SAM3.model.inference import *
from Evaluation_SAM3.utils.visualization import *
from Evaluation_SAM3.utils.operations import *

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test SAM3 with interactive bbox selection.")
    parser.add_argument("--image_path", type=str, default="", help="Path to the input image.")
    parser.add_argument("--THR", type=float, default=0.5, help="Score threshold for plotting boxes/masks.")
    parser.add_argument(
        "--model",
        type=str,
        default="/media/hassan/Data_vault1/datasets_hassan/models/sam3.pt",
        help="Path to the SAM3 model checkpoint.",
    )
    parser.add_argument(
        "--plot_bbox",
        action="store_true",
        help="Whether to plot bounding boxes along with masks.",
    )
    parser.add_argument(
        "--valstep",
        type=float,
        default=0.05,
        help="Step size for the threshold slider.",
    )

    args = parser.parse_args()
    image_path = args.image_path
    if image_path == "":
        image_path = "/home/hassan/Downloads/2A_media2/downloaded_images/frame_video_BYEDP220009_2025-10-21_133930_cam2_134938_742000_742000.jpg"



    THR = args.THR  # score threshold for plotting

    # 1) User draws prompt bbox
    bbox = get_single_bbox_interactive(image_path)
    print("Selected bbox:", bbox)

    # 2) Show prompt bbox on image
    plot_prompt_bbox(image_path, bbox)

    # 3) Feed bbox into SAM3
    predictor = load_SAM3_predictor(model=args.model)
    results = run_SAM3_predictor(
        predictor=predictor,
        image_path=image_path,
        bbox=bbox,
    )
    print("Results count:", results[0].boxes.shape[0] if results is not None else 0)

    # 4) Extract (masks, boxes, scores) if present
    masks, bboxes, scores = run_SAM3_visual_exempler_prompt(results)
    if bboxes is not None and scores is not None:
        plot_bboxes_with_scores_bar(
            image_path,
            masks=masks,
            bboxes=bboxes,
            scores=scores,
            thr=THR,
            valstep=args.valstep,
            title="Masks + boxes from results (filtered)",
            plot_bbox = args.plot_bbox
        )
    else:
        print("No (masks, boxes, scores) found in results.")
        print("If you have them from another stage, call:")
        print("plot_bboxes_with_scores_bar(image_path, masks, bboxes, scores, thr=0.5)")
