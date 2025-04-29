from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

class Passenger(BaseModel):
    Sex: int
    Age: float

app = FastAPI()

with open("titanic_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/predict")
def predict(passenger: Passenger):
    input_data = np.array([[passenger.Sex, passenger.Age]])
    prediction = model.predict(input_data)
    return {"Survived": int(prediction[0])}
