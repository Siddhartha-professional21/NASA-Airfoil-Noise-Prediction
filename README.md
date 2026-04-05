# NASA Airfoil Noise Prediction - Neural Surrogate Model

## Project Overview
A deep learning surrogate model that predicts aerodynamic noise levels from NASA airfoil wind tunnel data. Replaces expensive CFD simulations with a neural network that predicts in milliseconds.

**Dataset:** NASA Airfoil Self-Noise - 1503 real wind tunnel experiments

**Problem:** Regression - predict sound level (dB) from 5 physical parameters

**Best Model:** Deep Neural Network - RMSE: 3.02 | R2: 0.79

---

## Results Summary

| Model | RMSE | R2 |
|---|---|---|
| Linear Regression (baseline) | 4.67 | 0.51 |
| Shallow Neural Network | 4.25 | 0.60 |
| **Deep Neural Network** | **3.02** | **0.79** |
| Autoencoder + Deep NN | 5.47 | 0.33 |

Deep Neural Network achieved **35% lower RMSE** than baseline.

---

## Tech Stack
- Python, TensorFlow, Keras
- Scikit-learn, Pandas, NumPy
- MLflow (experiment tracking)
- FastAPI (model serving)
- Docker (containerization)
- GitHub Actions (CI/CD)
- Matplotlib, Seaborn

---

## How to Run

### Run API locally
pip install -r requirements.txt
uvicorn app:app --reload

### API Endpoints
- GET  /         - Check API status
- GET  /health   - Health check
- POST /predict  - Predict noise level

### Example Request
frequency: 2000, angle: 5.0, chord_length: 0.1524, velocity: 39.6, suction_thickness: 0.0028

### Example Response
predicted_sound_level_dB: 125.19, unit: decibels, model: Deep Neural Network, status: success

---

## Key Findings
- Deep Neural Network with Dropout and BatchNormalization outperformed all models
- Neural surrogate predicts in milliseconds vs hours for CFD simulation
- Autoencoder compression lost useful information - raw features performed better
- Frequency has strongest negative correlation with sound level (-0.39)

---

## Dataset
UCI ML Repository - NASA Airfoil Self-Noise
https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise
