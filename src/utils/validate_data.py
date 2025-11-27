# src/utils/validate_data.py
import pandas as pd
import re

def simple_validate_soccer_data(df: pd.DataFrame, require_target: bool = True):
    failed_checks = []

    required_columns = [
        "id","player_name","games","time","goals","assists","xA",
        "shots","key_passes","yellow_cards","red_cards","position",
        "team_title","npg","npxG","xGChain","xGBuildup","league","season"
    ]
    if require_target:
        required_columns.append("xG")

    for col in required_columns:
        if col not in df.columns:
            failed_checks.append(f"Missing column: {col}")
        elif df[col].isnull().any():
            failed_checks.append(f"Null values in column: {col}")

    # Numeric range checks
    numeric_ranges = {
        "games": (0, 60),
        "time": (0, 4000),
        "goals": (0, 60),
        "assists": (0, 30),
        "shots": (0, 300),
        "key_passes": (0, 300),
        "yellow_cards": (0, 25),
        "red_cards": (0, 5)
    }
    if require_target:
        numeric_ranges["xG"] = (0, 60)

    for col, (low, high) in numeric_ranges.items():
        if col in df.columns and not df[col].between(low, high).all():
            failed_checks.append(f"Values out of range in {col}")

    # Position checks
    if "position" in df.columns:
        valid_positions = ["GK","DF","MF","FW"]
        def map_position(pos):
            pos = str(pos)
            if "GK" in pos: return "GK"
            elif "D" in pos: return "DF"
            elif "M" in pos: return "MF"
            elif "F" in pos or "S" in pos: return "FW"
            return None
        df["position_clean"] = df["position"].apply(map_position)
        if df["position_clean"].isnull().any():
            failed_checks.append("Invalid position values")

    # League checks
    if "league" in df.columns:
        valid_leagues = ["PremierLeague","BundesLiga","LaLiga","ASerie","League1"]
        if not df["league"].isin(valid_leagues).all():
            failed_checks.append("Invalid league values")

    # Season checks
    if "season" in df.columns:
        def valid_season(s):
            s = str(s)
            return bool(re.match(r"^\d{4}/\d{4}$", s)) or bool(re.match(r"^\d{4}$", s))
        if not df["season"].apply(valid_season).all():
            failed_checks.append("Invalid season format")

    return len(failed_checks) == 0, failed_checks
