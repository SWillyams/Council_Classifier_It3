import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

app = FastAPI()


# Code borrowed from week 11:
artefacts = {}
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP: runs once, before the first request
    artefacts["model"] = joblib.load("../model/model.joblib")
    artefacts["vectoriser"] = joblib.load("../model/vectoriser.joblib")
    yield
    # SHUTDOWN: runs once when the server stops
    artefacts.clear()

class PredictionRequest(BaseModel):
    request_text: str

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
def predict(request: PredictionRequest):
    text = request.request_text
    #TODO data cleaning?, handle invalid requests, handle low confidence
    if request.text.strip() == "":
        raise HTTPException(status_code=422, detail="text must not be blank")

    vectorised_text = artefacts["vectoriser"].transform([request])

    prediction = artefacts["model"].predict(vectorised_text)[0]

    confidence = max(artefacts["model"].predict_proba(vectorised_text)[0])

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


# model = joblib.load("../model/model.joblib")
#vectoriser = joblib.load("../model/vectoriser.joblib")

# def predict(request: str):