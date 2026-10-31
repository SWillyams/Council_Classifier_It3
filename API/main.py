import joblib
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from API_lib import get_confidence_feedback

#local testing
LOCAL_TESTING = True

#To run:
# py -m uvicorn main:app --reload

#Parameters
LOW_CONFIDENCE=0.27
MED_CONFIDENCE=0.310289
HIGH_CONFIDENCE=0.375

#Set up app
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("helpdesk_api")

artefacts = {}
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model
    model_dir = Path(__file__).resolve().parent.parent / "model"
    artefacts["model"] = joblib.load(model_dir / "model.joblib")
    artefacts["vectoriser"] = joblib.load(model_dir / "vectoriser.joblib")
    yield
    # SHUTDOWN: runs once when the server stops
    artefacts.clear()

app = FastAPI(title="Council request classifier",lifespan=lifespan)

#local testing
if LOCAL_TESTING:
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(CORSMiddleware, 
                    allow_origins=['*'], 
                    allow_methods=['*']
                    )

class PredictionRequest(BaseModel):
    request_text: str

#From Week 12
def model_is_loaded():
    if "model" in artefacts and "vectoriser" in artefacts:
        return True
    return False

#FastAPI functions:
@app.get("/")
def read_root():
    return {"message":"Council Request Classifier"}

@app.get("/health")
def health():
    if not model_is_loaded():
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status":"ok"}

@app.get("/categories")
def categories():
    if not model_is_loaded():
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {
        "categories": artefacts["model"].classes_.tolist()
    }

@app.post("/predict")
def predict(request: PredictionRequest):
    """Make a category prediction from request text"""
    text = request.request_text

    #Error handling
    if text.strip() == "":
        raise HTTPException(status_code=422, detail="text must not be blank")
    if not model_is_loaded():
        raise HTTPException(status_code=503, detail="Model is not loaded")

    #Make predictions
    try:
        vectorised_text = artefacts["vectoriser"].transform([text])
        prediction = artefacts["model"].predict(vectorised_text)[0]
        confidence = max(artefacts["model"].predict_proba(vectorised_text)[0])

    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")

    confidence_rank, needs_review, review_reason = get_confidence_feedback(confidence)

    if vectorised_text.nnz == 0:
        needs_review = True
        confidence_rank = "very_low"
        review_reason = "no_known_words"
        prediction = "" #Override prediction as model will still return result
        confidence = 0        


    return {
        "request_text": request.request_text,
        "predicted_category": prediction,
        "confidence": round(float(confidence), 3),
        "confidence_rank":confidence_rank,
        "needs_review": needs_review,
        "review_reason":review_reason
    }


#Archived code:
#Test code from Geeks for geeks:
#@app.post("/greet")
#def greet_user(name: str):
#    return {"message": f"Hello, {name}!"}


# model = joblib.load("../model/model.joblib")
#vectoriser = joblib.load("../model/vectoriser.joblib")

# def predict(request: str):