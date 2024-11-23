from fastapi import APIRouter, File, UploadFile, HTTPException
from app.model.model_loader import load_trained_model
from app.model.preprocess import preprocess_image
import numpy as np

# Define the router
router = APIRouter()

# Load the model once when the API starts
MODEL_PATH = "trained_model.keras"
model = load_trained_model(MODEL_PATH)

@router.post("/predict", summary="Predict Crop Disease")
async def predict(file: UploadFile = File(...)):
    """
    Predict the crop disease from the uploaded image.
    Args:
        file (UploadFile): The uploaded image file.
    Returns:
        dict: Prediction result.
    """

    print("Received request for prediction.")
    print(f"Content type: {file.content_type}")
    try:
        image_data = await file.read()
        print(f"Received image: {file.filename}")

        preprocessed_image = preprocess_image(image_data)

        predictions = model.predict(preprocessed_image)
        
        predicted_class = np.argmax(predictions, axis=-1).tolist()

        return {"predicted_class": predicted_class, "probabilities": predictions.tolist()}
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error processing the image or making predictions.")
