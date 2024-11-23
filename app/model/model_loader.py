from tensorflow.keras.models import load_model

def load_trained_model(model_path: str):
    """
    Load the trained model from the specified path.
    Args:
        model_path (str): Path to the trained model file.
    Returns:
        keras.Model: The loaded model.
    """
    try:
        model = load_model(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e
