from src.inference.batch_predictor import BatchPredictor

predictor = BatchPredictor()

predictor.predict_folder(
    input_folder="sample_images",
    output_csv="outputs/predictions.csv",
)