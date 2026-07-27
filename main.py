import os
import joblib
import pandas as pd
import numpy as np
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


class Entry(BaseModel):
    mood: float
    created_at: str


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
    prediction = max(1, min(10, prediction))

    return {"predicted_mood": round(prediction, 1)}


@app.post("/sentiment")
def analyze_sentiment(request: SentimentRequest):
    if not request.text or not request.text.strip():
        return {"label": "neutral", "score": 0.0}

    result = sentiment_analyzer(request.text)[0]
    return {
        "label": result["label"].lower(),
        "score": round(result["score"], 3),
    }


@app.post("/anomalies")
def detect_anomalies(entries: list[Entry]):
    """
    Expects a list of entries like [{"mood": 5, "created_at": "2026-07-01"}, ...]
    ordered oldest to newest.
    """
    if len(entries) < 14:
        return {"anomalies": [], "message": "Not enough data yet to detect patterns.", "debug": {}}

    moods = np.array([e.mood for e in entries])
    dates = [e.created_at for e in entries]

    debug = {}
    unusual_days = []

    # 1. Single unusual days: z-score against a rolling 30-day window.
    window = min(30, len(moods))
    for i in range(window, len(moods)):
        recent = moods[max(0, i - window):i]
        mean = recent.mean()
        std = recent.std()
        if std > 0:
            z = (moods[i] - mean) / std
            if abs(z) >= 2.5:
                unusual_days.append({
                    "type": "unusual_day",
                    "date": dates[i],
                    "direction": "low" if z < 0 else "high",
                    "z_score": round(float(z), 2),
                    "message": f"{dates[i][:10]} was unusually {'low' if z < 0 else 'high'} compared to your recent pattern."
                })

    unusual_days = sorted(unusual_days, key=lambda a: a["date"], reverse=True)[:3]
    anomalies = list(unusual_days)

    # 2. Sustained shift: last 14 days average vs. overall baseline
    if len(moods) >= 30:
        baseline = moods[:-14].mean()
        recent_avg = moods[-14:].mean()
        diff = recent_avg - baseline
        debug["sustained_shift"] = {
            "baseline_avg": round(float(baseline), 2),
            "recent_14d_avg": round(float(recent_avg), 2),
            "difference": round(float(diff), 2),
            "threshold": 1.5,
            "triggered": bool(abs(diff) >= 1.5),
        }
        if abs(diff) >= 1.5:
            anomalies.append({
                "type": "sustained_shift",
                "direction": "low" if diff < 0 else "high",
                "message": f"Your average mood over the last 2 weeks has been notably {'lower' if diff < 0 else 'higher'} than your typical baseline."
            })

    # 3. Increased volatility: recent rolling std vs. overall std
    if len(moods) >= 30:
        overall_std = moods[:-14].std()
        recent_std = moods[-14:].std()
        ratio = recent_std / overall_std if overall_std > 0 else 0
        debug["volatility"] = {
            "overall_std": round(float(overall_std), 2),
            "recent_14d_std": round(float(recent_std), 2),
            "ratio": round(float(ratio), 2),
            "threshold": 1.5,
            "triggered": bool(overall_std > 0 and ratio >= 1.5),
        }
        if overall_std > 0 and ratio >= 1.5:
            anomalies.append({
                "type": "volatility",
                "message": "Your mood has been swinging more than usual over the last couple weeks."
            })

    return {"anomalies": anomalies, "debug": debug}