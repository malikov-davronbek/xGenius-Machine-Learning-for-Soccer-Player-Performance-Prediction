import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.serving.inference import InferencePipeline



pipeline = InferencePipeline(
    model_path=os.path.join("models", "final_best_gb_model.joblib"),
    features_path=os.path.join("models", "feature_names.joblib")
)

sample_input = {
    "id": 101,
    "player_name": "Erling Haaland",
    "games": 35,
    "time": 3000,
    "goals": 32,
    "assists": 5,
    "xA": 4.2,
    "shots": 120,
    "key_passes": 25,
    "yellow_cards": 3,
    "red_cards": 0,
    "position": "FW",
    "team_title": "Manchester City",
    "npg": 29.0,
    "npxG": 27.5,
    "xGChain": 45.2,
    "xGBuildup": 12.3,
    "league": "ASerie",
    "season": 2023
}

result = pipeline.predict(sample_input)
print(result)
