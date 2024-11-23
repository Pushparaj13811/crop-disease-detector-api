from fastapi import FastAPI
from app.routes.predict import router as predict_router

app = FastAPI(title="Crop Disease Classification API")

# Include the predict route
app.include_router(predict_router)

@app.get("/")
def home():
    return {"message": "Welcome to the Crop Disease Classification API"}
