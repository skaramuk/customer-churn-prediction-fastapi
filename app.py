from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# --------------------
# MODEL LOAD
# --------------------
with open("churn_gradient_boosting_model.pkl", "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    scaler = data["scaler"]
    feature_names = data["feature_names"]

# --------------------
# INPUT SCHEMA
# --------------------
class ChurnFeatures(BaseModel):
    Customer_Age: int
    Dependent_count: int
    Education_Level: float
    Income_Category: float
    Months_on_book: int
    Total_Relationship_Count: int
    Months_Inactive_12_mon: int
    Contacts_Count_12_mon: int
    Credit_Limit: float
    Total_Revolving_Bal: int
    Avg_Open_To_Buy: float
    Total_Amt_Chng_Q4_Q1: float
    Total_Trans_Amt: int
    Total_Trans_Ct: int
    Total_Ct_Chng_Q4_Q1: float
    Avg_Utilization_Ratio: float
    Gender_M: int
    Marital_Status_Married: int
    Marital_Status_Single: int
    Marital_Status_Unknown: int
    Card_Category_Gold: int
    Card_Category_Platinum: int
    Card_Category_Silver: int

# --------------------
# ENDPOINT
# --------------------
@app.post("/predict")
async def predict(features: ChurnFeatures):
    try:
        input_df = pd.DataFrame([features.model_dump()])


        input_df = input_df[feature_names]

        # scale
        input_scaled = scaler.transform(input_df)

        # prediction
        churn_prob = model.predict_proba(input_scaled)[0, 1]
        churn_pred = int(churn_prob >= 0.5)

        return {
            "churn_prediction": churn_pred,
            "churn_probability": round(float(churn_prob), 4)
        }

    except Exception as e:
        return {"error": str(e)}

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()