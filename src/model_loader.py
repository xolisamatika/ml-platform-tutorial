import os
import pickle
import mlflow


def load_model():
    source = os.getenv("MODEL_SOURCE", "mlflow")

    if source == "local":
        print("Loading model from local pickle file...")
        with open("models/model.pkl", "rb") as f:
            model, encoder = pickle.load(f)

        return model, encoder

    elif source == "mlflow":
        print("Loading model from MLflow Model Registry...")

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

        try:
            model = mlflow.sklearn.load_model("models:/fraud-detection-model@champion")
            print("Successfully loaded champion model from MLflow!")
        
            # with open("encoder.pkl", "rb") as f:
            #     encoder = pickle.load(f)
            # print("Encoder loaded successfully!")
            return model, None
        except Exception as e:
            print(f"Error loading from MLflow: {e}")
            print("Make sure you've assigned the @champion alias to a model in the MLflow UI")
            raise
    else:
        raise ValueError(f"Unknown MODEL_SOURCE: {source}")