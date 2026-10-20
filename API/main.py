import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("../model/model.joblib")
vectoriser = joblib.load("../model/vectoriser.joblib")

# class PredictionRequest(BaseModel):
#     request_text: str

@app.get("/")
def read_root():
    return {"message":"Council Request Classifier"}

@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/categories")
def categories():
    return {
        "categories": model.classes_.tolist()
    }

@app.post("/predict")
#def predict(request: PredictionRequest):
#    text = request.request_text
#    vectorised_text = vectoriser.transform([text])

def predict(request: str):
    #TODO data cleaning?, handle invalid requests, handle low confidence
    vectorised_text = vectoriser.transform([request])

    prediction = model.predict(vectorised_text)[0]

    confidence = max(model.predict_proba(vectorised_text)[0])

    return {
        "request_text": request,
        "predicted_category": prediction,
        "confidence": round(float(confidence), 3)
    }


#Archived code:
#Test code from Geeks for geeks:
#@app.post("/greet")
#def greet_user(name: str):
#    return {"message": f"Hello, {name}!"}