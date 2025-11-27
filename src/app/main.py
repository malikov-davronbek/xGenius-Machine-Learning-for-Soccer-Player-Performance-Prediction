# src/app/main.py
import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel

# ─── Make project root importable ─────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.serving.inference import InferencePipeline

# Load model + pipeline
MODEL_PATH = os.path.join(ROOT_DIR, "models", "final_best_gb_model.joblib")
FEATURES_PATH = os.path.join(ROOT_DIR, "models", "feature_names.joblib")
pipeline = InferencePipeline(model_path=MODEL_PATH, features_path=FEATURES_PATH)

# Initialize FastAPI
app = FastAPI(
    title="Soccer xG Prediction API",
    description="Predict expected goals per game using raw player stats",
    version="1.0"
)

# Health check
@app.get("/")
def root():
    return {"status": "ok"}

# Raw input schema
class PlayerData(BaseModel):
    id: int
    player_name: str
    games: int
    time: float
    goals: int
    assists: int
    xA: float
    shots: int
    key_passes: int
    yellow_cards: int
    red_cards: int
    position: str
    team_title: str
    npg: float
    npxG: float
    xGChain: float
    xGBuildup: float
    league: str
    season: int

# Prediction endpoint
@app.post("/predict")
def predict(data: PlayerData):
    raw_features = data.dict()
    result = pipeline.predict(raw_features)
    return {"prediction": result}
