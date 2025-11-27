# src/app/app.py
import os
import sys
import gradio as gr
from fastapi import FastAPI

# ─── Make project root importable ─────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.serving.inference import InferencePipeline

# Load model + pipeline
MODEL_PATH = os.path.join(ROOT_DIR, "models", "final_best_gb_model.joblib")
FEATURES_PATH = os.path.join(ROOT_DIR, "models", "feature_names.joblib")
pipeline = InferencePipeline(model_path=MODEL_PATH, features_path=FEATURES_PATH)

# Gradio prediction function
def gradio_ui(
    id, player_name, games, time, goals, assists, xA, shots,
    key_passes, yellow_cards, red_cards, position, team_title,
    npg, npxG, xGChain, xGBuildup, league, season
):
    payload = {
        "id": id,
        "player_name": player_name,
        "games": games,
        "time": time,
        "goals": goals,
        "assists": assists,
        "xA": xA,
        "shots": shots,
        "key_passes": key_passes,
        "yellow_cards": yellow_cards,
        "red_cards": red_cards,
        "position": position,
        "team_title": team_title,
        "npg": npg,
        "npxG": npxG,
        "xGChain": xGChain,
        "xGBuildup": xGBuildup,
        "league": league,
        "season": season
    }
    return pipeline.predict(payload)

# Gradio input widgets
inputs = [
    gr.Number(label="ID"),
    gr.Textbox(label="Player Name"),
    gr.Number(label="Games"),
    gr.Number(label="Time"),
    gr.Number(label="Goals"),
    gr.Number(label="Assists"),
    gr.Number(label="xA"),
    gr.Number(label="Shots"),
    gr.Number(label="Key Passes"),
    gr.Number(label="Yellow Cards"),
    gr.Number(label="Red Cards"),
    gr.Textbox(label="Position"),
    gr.Textbox(label="Team Title"),
    gr.Number(label="NPG"),
    gr.Number(label="NPxG"),
    gr.Number(label="xGChain"),
    gr.Number(label="xGBuildup"),
    gr.Textbox(label="League"),
    gr.Number(label="Season"),
]

# Create Gradio interface
demo = gr.Interface(
    fn=gradio_ui,
    inputs=inputs,
    outputs="json",
    title="Soccer Player xG Predictor",
)

# Initialize FastAPI and mount Gradio
app = FastAPI(
    title="Soccer xG Prediction API",
    description="Predict expected goals per game from raw player stats",
    version="1.0"
)
app = gr.mount_gradio_app(app, demo, path="/ui")
