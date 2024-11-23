from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image
import io

def preprocess_image(image):
    """
    Preprocess the uploaded image for model prediction.
    Args:
        image (file-like or str): The uploaded image in bytes or file path format.
    Returns:
        np.ndarray: Preprocessed image ready for model prediction.
    """
    try:
        # If the input is in bytes format (as received from frontend), convert it to a PIL image
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, str):  # If the input is a file path
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):  # If the input is already a PIL image
            image = image.convert("RGB")
        else:
            raise ValueError("Input must be a file path, a PIL Image object, or bytes.")

        # Resize image to (128, 128) as required by your model
        image = image.resize((128, 128))

        # Convert image to array
        input_arr = img_to_array(image)

        # Add batch dimension (1, 128, 128, 3)
        input_arr = np.expand_dims(input_arr, axis=0)

        return input_arr

    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise e
