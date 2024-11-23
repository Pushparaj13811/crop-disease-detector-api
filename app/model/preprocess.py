from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from io import BytesIO
from PIL import Image

def preprocess_image(image: bytes):
    """
    Preprocess the uploaded image for model prediction.
    Args:
        image (bytes): The uploaded image in bytes format.
    Returns:
        np.ndarray: Preprocessed image ready for model prediction.
    """
    try:
        # Load image using PIL
        pil_image = Image.open(BytesIO(image)).convert("RGB")
        
        # Resize image to 128x128 (required by your model)
        pil_image = pil_image.resize((128, 128))
        
        # Convert image to array
        image_array = img_to_array(pil_image)
        
        # Add batch dimension (1, 128, 128, 3)
        image_array = np.expand_dims(image_array, axis=0)
        
        # Preprocess the image (normalization, etc.)
        image_array = preprocess_input(image_array)
        
        return image_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise e
