# Crop Disease Detector API

This repository provides an API built with **FastAPI** for detecting crop diseases using a pretrained deep learning model. The model is trained to recognize various crop diseases from images, making it a valuable tool for agricultural purposes. It uses a convolutional neural network (CNN) to predict the class of the disease based on the input image.

## Features

- **Disease Prediction**: Classify images of crops and predict the type of disease affecting them.
- **Pretrained Model**: The model has been trained on a large dataset of crop disease images.
- **FastAPI Backend**: The API is built using FastAPI, which is known for its high performance.
- **Easy Deployment**: Docker support for easy deployment and containerization.

## Requirements

- Python 3.12 or higher
- FastAPI
- TensorFlow
- Uvicorn
- Docker (for containerization)

## Installation

### Clone the Repository

```bash
git clone https://github.com/Pushparaj13811/crop-disease-detector-api.git
cd crop-disease-detector-api
```

### Install Dependencies

Create a virtual environment and install the required dependencies using the following commands:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

### Running the API Locally

To run the FastAPI application locally, use Uvicorn as the server:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

### Docker Setup (Optional)

You can also run the application in a Docker container for easy deployment:

1. **Build the Docker image:**

   ```bash
   docker build -t crop-disease-detector-api .
   ```

2. **Run the container:**

   ```bash
   docker run -p 8000:8000 crop-disease-detector-api
   ```

3. **Access the API**: Visit `http://localhost:8000` in your browser.

## API Endpoints

### `POST /predict`

This endpoint takes an image of a crop and returns the predicted disease class along with the probabilities for each class.

**Request**:
- Content-Type: `multipart/form-data`
- Body: Image file (JPEG, PNG formats are supported)

Example Request (using `curl`):
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -F 'file=@your_image.jpg'
```

**Response**:

```json
{
  "predicted_class": "Corn_(maize)___Common_rust_",
  "predicted_crop": "Corn_(maize)",
  "predicted_diseases": "Common_rust_",
  "confidence_percentage": 100
}
```

The `predicted_class` indicates the disease category, and `probabilities` provides the confidence level for each possible class.

## Model Information

The model used in this API is a **pretrained deep learning model** (e.g., a CNN model like ResNet, VGG, etc.) fine-tuned to recognize crop diseases. It has been trained on a variety of crop disease datasets and is capable of predicting different diseases affecting crops.

## Contributing

Contributions are welcome! If you would like to contribute to the development of this project, feel free to fork the repository, create a branch, and submit a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Create a new pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **TensorFlow**: For providing the framework for training and deploying machine learning models.
- **FastAPI**: For providing a high-performance web framework for building APIs.
- **Docker**: For enabling easy deployment of the application in isolated containers.
