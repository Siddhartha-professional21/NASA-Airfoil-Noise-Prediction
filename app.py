import numpy as np
import pickle
import tensorflow as tf
from fastapi  import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title       = "NASA Airfoil Noise Predictor",
    description = "Predicts aerodynamic noise using Deep Learning",
    version     = "1.0.0"
)

model  = tf.keras.models.load_model("models/best_model.keras")

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

class AirfoilInput(BaseModel):
    frequency         : float
    angle             : float
    chord_length      : float
    velocity          : float
    suction_thickness : float

@app.get("/")
def home():
    return {"message" : "NASA Airfoil Noise Predictor API is running"}

@app.get("/health")
def health():
    return {"status" : "healthy"}

@app.post("/predict")
def predict(data: AirfoilInput):
    new_input        = np.array([[data.frequency, data.angle,
                                  data.chord_length, data.velocity,
                                  data.suction_thickness]])
    new_input_scaled = scaler.transform(new_input)
    prediction       = model.predict(new_input_scaled, verbose=0)
    result           = round(float(prediction[0][0]), 2)
    return {
        "predicted_sound_level_dB" : result,
        "unit"                     : "decibels",
        "model"                    : "Deep Neural Network",
        "status"                   : "success"
    }
