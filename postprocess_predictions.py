import requests
import pandas as pd
import numpy as np

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


def fetch_availability_map():
    """
    Returns dict:
    element_id -> chance_of_playing_next_round
    """
    r = requests.get(BOOTSTRAP_URL, timeout=10)
    r.raise_for_status()
    data = r.json()

    availability = {}

    for p in data["elements"]:
        availability[p["id"]] = {
            "chance": p.get("chance_of_playing_next_round"),
            "status": p.get("status")
        }

    return availability


def apply_availability_rule(df_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Set predicted points to 0 for injured/suspended/unavailable players
    who have 0% chance of playing next GW.
    """
    availability = fetch_availability_map()

    df = df_pred.copy()

    for idx, row in df.iterrows():
        elem = row["element"]

        info = availability.get(elem)
        if not info:
            continue

        chance = info["chance"]
        status = info["status"]

        # Only zero if confirmed OUT
        if chance == 0 and status in ["i", "s", "u"]:
            df.at[idx, "pred_points"] = 0.0

    return df


def apply_integer_rule(df_pred: pd.DataFrame) -> pd.DataFrame:
    df_pred["pred_points"] = np.floor(df_pred["pred_points"]).astype(int)
    return df_pred
