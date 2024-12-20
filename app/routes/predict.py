from fastapi import APIRouter, File, UploadFile, HTTPException
from app.model.model_loader import load_trained_model
from app.model.preprocess import preprocess_image
import numpy as np

# Define the router
router = APIRouter()

# Load the model once when the API starts
MODEL_PATH = "trained_model.keras"
model = load_trained_model(MODEL_PATH)

# Define class names
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

@router.post("/predict", summary="Predict Crop Disease")
async def predict(file: UploadFile = File(...)):
    """
    Predict the crop disease from the uploaded image.
    Args:
        file (UploadFile): The uploaded image file.
    Returns:
        dict: Prediction result with predicted class, crop, disease, and confidence percentage.
    """
    
    try:
        image_data = await file.read()

        preprocessed_image = preprocess_image(image_data)

        predictions = model.predict(preprocessed_image)

        predicted_class_idx = np.argmax(predictions, axis=-1).tolist()

        confidence = predictions[0][predicted_class_idx] * 100

        predicted_class = class_names[predicted_class_idx[0]]

        predicted_crop, predicted_disease = predicted_class.split("___")

        confidence = float(confidence)
        
        if predicted_disease == "healthy":
            return{
                "predicted_class": predicted_class,
                "predicted_crop": predicted_crop,
                "isHealthy": "Healthy",
                "predicted_diseases": "Null",
                "confidence_percentage": confidence              
            }
        else:
            diseases = predicted_disease


        # Return the formatted response
        return {
            "predicted_class": predicted_class,
            "predicted_crop": predicted_crop,
            "isHealthy": "Unhealthy",
            "predicted_diseases": diseases,
            "confidence_percentage": confidence
        }

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error processing the image or making predictions.")
