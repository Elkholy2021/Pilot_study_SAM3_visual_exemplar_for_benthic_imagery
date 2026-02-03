from ultralytics.models.sam import SAM3SemanticPredictor

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