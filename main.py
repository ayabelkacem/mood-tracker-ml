import os
import joblib
import pandas as pd
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model once when the service starts
model = joblib.load("mood_model.joblib")

# Load a small pretrained sentiment analysis model once at startup
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


class PredictionRequest(BaseModel):
    sleep: float
    recent_mood_avg: float
    has_work_tag: bool = False
    has_health_tag: bool = False
    has_social_tag: bool = False


class SentimentRequest(BaseModel):
    text: str


@app.get("/")
def read_root():
    return {"status": "ML service is running"}


@app.post("/predict")
def predict_mood(request: PredictionRequest):
    day_of_week = datetime.now().weekday()

    features = pd.DataFrame([{
        "sleep": request.sleep,
        "day_of_week": day_of_week,
        "has_work_tag": int(request.has_work_tag),
        "has_health_tag": int(request.has_health_tag),
        "has_social_tag": int(request.has_social_tag),
        "recent_mood_avg": request.recent_mood_avg,
    }])

    prediction = model.predict(features)[0]
    # Clamp between 1 and 10, since mood scores are on that scale
    prediction = max(1, min(10, prediction))

    return {"predicted_mood": round(prediction, 1)}


@app.post("/sentiment")
def analyze_sentiment(request: SentimentRequest):
    if not request.text or not request.text.strip():
        return {"label": "neutral", "score": 0.0}

    result = sentiment_analyzer(request.text)[0]
    return {
        "label": result["label"].lower(),  # "positive" or "negative"
        "score": round(result["score"], 3),  # confidence, 0 to 1
    }