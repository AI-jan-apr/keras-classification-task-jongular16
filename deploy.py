from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI(title="Breast Cancer Classification API")

with open("scaler_weights.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model_weights.pkl", "rb") as f:
    model = pickle.load(f)

latest_input = None

FEATURE_COLUMNS = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]


class CancerInput(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float


@app.post("/input")
def take_input(data: CancerInput):
    global latest_input

    payload = data.model_dump()

    latest_input = {
        "radius_mean": payload["radius_mean"],
        "texture_mean": payload["texture_mean"],
        "perimeter_mean": payload["perimeter_mean"],
        "area_mean": payload["area_mean"],
        "smoothness_mean": payload["smoothness_mean"],
        "compactness_mean": payload["compactness_mean"],
        "concavity_mean": payload["concavity_mean"],
        "concave points_mean": payload["concave_points_mean"],
        "symmetry_mean": payload["symmetry_mean"],
        "fractal_dimension_mean": payload["fractal_dimension_mean"],
        "radius_se": payload["radius_se"],
        "texture_se": payload["texture_se"],
        "perimeter_se": payload["perimeter_se"],
        "area_se": payload["area_se"],
        "smoothness_se": payload["smoothness_se"],
        "compactness_se": payload["compactness_se"],
        "concavity_se": payload["concavity_se"],
        "concave points_se": payload["concave_points_se"],
        "symmetry_se": payload["symmetry_se"],
        "fractal_dimension_se": payload["fractal_dimension_se"],
        "radius_worst": payload["radius_worst"],
        "texture_worst": payload["texture_worst"],
        "perimeter_worst": payload["perimeter_worst"],
        "area_worst": payload["area_worst"],
        "smoothness_worst": payload["smoothness_worst"],
        "compactness_worst": payload["compactness_worst"],
        "concavity_worst": payload["concavity_worst"],
        "concave points_worst": payload["concave_points_worst"],
        "symmetry_worst": payload["symmetry_worst"],
        "fractal_dimension_worst": payload["fractal_dimension_worst"],
    }

    return {
        "message": "Input stored successfully",
        "stored_input": latest_input
    }


@app.get("/predict")
def get_prediction():
    global latest_input

    if latest_input is None:
        raise HTTPException(
            status_code=400,
            detail="No input found. Send data first using POST /input"
        )

    try:
        input_df = pd.DataFrame([latest_input])
        input_df = input_df[FEATURE_COLUMNS]

        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)

        prob = float(prediction[0][0]) if len(prediction.shape) > 1 else float(prediction[0])

        predicted_class = 1 if prob >= 0.5 else 0
        predicted_label = "Malignant" if predicted_class == 1 else "Benign"

        return {
            "prediction": predicted_label,
            "prediction_code": predicted_class,
            "probability_malignant": prob,
            "probability_benign": 1 - prob
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))