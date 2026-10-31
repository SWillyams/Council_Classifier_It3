Council Request Classifier Project for BCIT601

API uses FastAPI, joblib and Scikit-learn. See API/requirements.txt for exact requirements.
To run:
Run server with "py -m uvicorn main:app --reload" from API folder

To view API endpoints Open localhost:8000/docs

To test predict endpoint with a browser ensure LOCAL_TESTING is set to true in api\main.py and
open use_predict_api.html

Machine learning model has been extensively tested in training folder.

Tests:
run "python -m pytest -q" in terminal from project root folder

Model training:
To retrain the model, open training/ModelTraining.py
Ensure model_type is selected and run_final_model_training is True; then run file
Additional parameters can be set, such as new training data. Parameters and defaults are listed
under if run_final_model_training.
Copy vectoriser.joblib and model.joblib files over to model folder and ensure they are named as such.