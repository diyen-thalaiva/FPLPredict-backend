# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import requests
from datetime import datetime, timezone,timedelta

from inference_core import predict_next_gw_pipeline
from postprocess_predictions import (
    apply_availability_rule,
    apply_integer_rule,
)
from fpl_bootstrap import enrich_predictions_with_bootstrap, get_player_enrichment_map
# -------------------------------------------------
# App setup
# -------------------------------------------------
app = FastAPI(
    title="FPLPredict API",
    description="Prototype ML-powered FPL prediction service",
    version="0.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # prototype only
    allow_methods=["*"],
    allow_headers=["*"],
)

PLAYER_ID_COL = "name"

# -------------------------------------------------
# Load data ONCE at startup
# -------------------------------------------------
print("🔄 Loading datasets...")

# Training seasons (ONLY to identify known players)
df_all = pd.read_parquet("artifacts/df_all.parquet")
KNOWN_PLAYERS = set(df_all["name"].dropna().unique())

# 2025–26 engineered features (prediction input)
df_2526 = pd.read_csv("artifacts/df_2526_feature_engineered.csv")

print("✅ Known players:", len(KNOWN_PLAYERS))
print("✅ 2025–26 rows:", df_2526.shape)

# -------------------------------------------------
# FPL API endpoints
# -------------------------------------------------
FPL_ENTRY_PICKS = "https://fantasy.premierleague.com/api/entry/{manager_id}/event/{gw}/picks/"
FPL_ENTRY_HISTORY = "https://fantasy.premierleague.com/api/entry/{manager_id}/history/"

#Fallback helper
def fetch_picks_with_fallback(manager_id: int, gw: int):
    r = requests.get(FPL_ENTRY_PICKS.format(manager_id=manager_id, gw=gw), timeout=10)
    if r.status_code == 200:
        return r.json(), gw

    prev_gw = gw - 1
    if prev_gw < 1:
        raise HTTPException(status_code=404, detail="No previous GW available")

    r_prev = requests.get(FPL_ENTRY_PICKS.format(manager_id=manager_id, gw=prev_gw), timeout=10)
    r_prev.raise_for_status()
    return r_prev.json(), prev_gw


def get_current_or_next_gw():
    """Determines the target GW based on the current real-world FPL status."""
    try:
        r = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=10)
        r.raise_for_status()
        data = r.json()
        events = data.get('events', [])
        
        current_gw = next((e for e in events if e['is_current']), None)
        next_gw = next((e for e in events if e['is_next']), None)

        if current_gw and not current_gw['finished']:
            return current_gw['id']
        elif next_gw:
            return next_gw['id']
        return 1
    except Exception:
        return 1

def is_deadline_passed(gw: int):
    """
    Returns True if the deadline for the given GW has passed.
    """
    try:
        r = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=10)
        r.raise_for_status()
        data = r.json()
        events = data.get("events", [])

        event = next((e for e in events if e["id"] == gw), None)
        if not event:
            return False

        deadline_str = event.get("deadline_time")
        if not deadline_str:
            return False

        deadline = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)

        return now >= deadline

    except Exception:
        return False
    

TEAM_MAP = {}

def get_team_mapping():
    global TEAM_MAP
    if TEAM_MAP:
        return TEAM_MAP

    res = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
    data = res.json()

    TEAM_MAP = {team["id"]: team["short_name"] for team in data["teams"]}
    return TEAM_MAP

def format_datetime(iso_time):

    if not iso_time:
        return {
            "day": None,
            "date": None,
            "time": "TBD",
            "display_date": "TBD"
        }

    dt_utc = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
    # 2. Convert to Mauritius Time (UTC +4)
    dt_mu = dt_utc + timedelta(hours=4)
    return {
        "day": dt_mu.strftime("%A"),
        "date": dt_mu.strftime("%B %d, %Y"),
        "time": dt_mu.strftime("%H:%M"),
        "display_date": dt_mu.strftime("%A, %B %d, %Y")
    }
# -------------------------------------------------
# 🔄 FREE TRANSFERS + BANK (PLANNER SAFE)
# -------------------------------------------------

def calculate_free_transfers_live(manager_id: int, target_gw: int) -> int:
    """
    2025/26 Rules-based simulation using History API:
    - Max 5 banked transfers.
    - GW16 AFCON Top-up to 5.
    - Wildcard/Free Hit preserves banked status.
    """
    try:
        r = requests.get(FPL_ENTRY_HISTORY.format(manager_id=manager_id), timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"Error fetching history for FT calculation: {e}")
        return 1 # Fallback to 1 if API fails

    history = data.get('current', [])
    chips = {c['event']: c['name'] for c in data.get('chips', [])}
    
    banked_transfers = 0  # Starts at 0 after GW1 is picked
    
    # Iterate through completed Gameweeks present in history
    for gw in history:
        event = gw['event']
        
        # 1. How many did we start THIS week with?
        # (Banked from last week + 1 new transfer, capped at 5)
        available_this_week = min(5, banked_transfers + 1)
        
        # 2. Rule: GW16 Reset
        if event == 16:
            available_this_week = 5
            
        # 3. How many free ones were consumed?
        transfers_made = gw.get('event_transfers', 0)
        hits_taken = gw.get('event_transfers_cost', 0) // 4
        free_used = max(0, transfers_made - hits_taken)
        
        # 4. Determine what carries over to NEXT week
        chip_played = chips.get(event)
        if chip_played in ['wildcard', 'freehit']:
            # In 25/26, these chips preserve the balance you HAD
            new_banked = banked_transfers 
        else:
            new_banked = max(0, available_this_week - free_used)
        
        banked_transfers = new_banked

    # The result for the 'target_gw' (the one being planned) 
    # is the banked amount from history + the 1 new transfer for the target week.
    final_available = min(5, banked_transfers + 1)
    
    # Special case: If target_gw is 16, force 5
    if target_gw == 16:
        return 5
        
    return final_available

def get_bank_live(manager_id: int, target_gw: int) -> float:
    """
    Returns bank from previous GW (planner safe)
    """

    source_gw = target_gw - 1

    if source_gw < 1:
        return 0.0

    r = requests.get(
        FPL_ENTRY_PICKS.format(manager_id=manager_id, gw=source_gw),
        timeout=10
    )
    r.raise_for_status()

    data = r.json()
    bank = data.get("entry_history", {}).get("bank", 0)

    return bank / 10  # convert to £m
# -------------------------------------------------
# Health check
# -------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "FPLPredict prototype API running"}

# -------------------------------------------------
# 🌍 League-wide predictions (optional demo)
# -------------------------------------------------
@app.get("/predict/next-gw/{gw}")
def predict_next_gw(gw: int):
    
    df_pred = predict_next_gw_pipeline(
        pipeline_path="artifacts/fpl_xgb_pipeline.pkl",
        feature_list_path="artifacts/feature_list.json",
        df_full=df_2526,
        season="2025-26",
        gw_to_predict=gw,
        known_players=KNOWN_PLAYERS,
        player_id_col=PLAYER_ID_COL,
    )

    if df_pred is None or df_pred.empty:
        raise HTTPException(status_code=404, detail="No predictions available")

    # Enrich with bootstrap data (web_name, news)
    df_pred = enrich_predictions_with_bootstrap(df_pred,gw)  # ← ADD THIS
    
    df_pred = apply_availability_rule(df_pred)
    df_pred = apply_integer_rule(df_pred)

    return {
        "season": "2025-26",
        "gw": gw,
        "count": len(df_pred),
        "predictions": df_pred[
            [
                "name",
                "web_name",           
                "position",
                "value",
                "team_name",
                "opponent_name",
                "fixture_1_5_fdr",
                "pred_points",
                "form",
                "ownership_pct",
                "net_transfer_pct",
                "news",  
                "fixtures",
                "next_3_fdrs",
                "total_points"

            ]
        ].to_dict(orient="records"),
    }

# -------------------------------------------------
# ⭐ PREDICTION ENDPOINT FOR PREDICTION PAGE (CORE PROTOTYPE)
# -------------------------------------------------
@app.get("/manager/{manager_id}/prediction")
def manager_prediction(manager_id: int):
    """
    Returns predicted points for a manager lineup for target GW.
    If GW lineup unavailable, automatically uses previous GW lineup.
    """
    target_gw = get_current_or_next_gw()

    # 1) Fetch manager data (Picks, Info, and History)
    try:
        # Get picks and active chip for this specific GW
        picks_data, team_source_gw = fetch_picks_with_fallback(manager_id, target_gw)
        picks = picks_data.get("picks", []) # Original list with squad positions (1-15)
        if team_source_gw < target_gw:
            active_chip = None
        else:
            active_chip = picks_data.get("active_chip")

        # Get manager info (Names)
        r_manager = requests.get(f"https://fantasy.premierleague.com/api/entry/{manager_id}/", timeout=10)
        r_manager.raise_for_status()
        manager_info = r_manager.json()
        
        # Get chip history to determine availability
        r_history = requests.get(FPL_ENTRY_HISTORY.format(manager_id=manager_id), timeout=10)
        r_history.raise_for_status()
        history_data = r_history.json()
        raw_chips_used = history_data.get("chips", []) 
        
        chips_used = [
            {"name": str(c['name']), "gw": int(c['event'])} 
            for c in raw_chips_used
        ]

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"FPL API error: {e}")

    if not picks:
        raise HTTPException(status_code=404, detail="No picks found for this manager/GW")

    # --- CHIP RESET LOGIC (2025/26 RULES) ---
    all_chips_set = {"wildcard", "3xc", "bboost", "freehit"}
    
    if target_gw < 20:
        used_in_window = {str(c['name']) for c in chips_used if int(c['gw']) < 20}
    else:
        used_in_window = {str(c['name']) for c in chips_used if int(c['gw']) >= 20}

    available_chips = list(all_chips_set - used_in_window)

    if active_chip and active_chip in available_chips:
        available_chips.remove(active_chip)

    manager_name = f"{manager_info.get('player_first_name', '')} {manager_info.get('player_last_name', '')}".strip()
    team_name = manager_info.get('name', '')
    
    # Build pick mapping for quick lookup
    pick_map = {
        pick["element"]: {
            "is_captain": bool(pick.get("is_captain", False)),
            "is_vice_captain": bool(pick.get("is_vice_captain", False)),
            "multiplier": int(pick.get("multiplier", 1)),
            "squad_position": int(pick.get("position", 0)) # 1-11 are starters, 12-15 bench
        } for pick in picks
    }

    # --- Reset multipliers if we fell back to a previous week ---
    if team_source_gw < target_gw:
        for elem in pick_map:
            if pick_map[elem]["is_captain"]:
                pick_map[elem]["multiplier"] = 2
            else:
                pick_map[elem]["multiplier"] = 1

    # Vice-Captain Promotion Logic
    captain_elem = None
    vice_elem = None

    for elem, info in pick_map.items():
        if info["is_captain"]:
            captain_elem = elem
        if info["is_vice_captain"]:
            vice_elem = elem

    if captain_elem:
        captain_on_bench = pick_map[captain_elem]["squad_position"] > 11
        if captain_on_bench and vice_elem:
            pick_map[captain_elem]["multiplier"] = 1
            pick_map[vice_elem]["multiplier"] = 2

    # 2) Run ML Prediction Pipeline
    df_pred = predict_next_gw_pipeline(
        pipeline_path="artifacts/fpl_xgb_pipeline.pkl",
        feature_list_path="artifacts/feature_list.json",
        df_full=df_2526,
        season="2025-26",
        gw_to_predict=target_gw,
        known_players=KNOWN_PLAYERS,
        player_id_col=PLAYER_ID_COL,
    )

    if df_pred is None or df_pred.empty:
        raise HTTPException(status_code=500, detail="Prediction failed")

    df_pred = enrich_predictions_with_bootstrap(df_pred, target_gw)
    df_pred = apply_availability_rule(df_pred)
    df_pred = apply_integer_rule(df_pred)

    # --- ADDED: Fetch Bootstrap Data for "Blank" Player Lookups ---
    bootstrap_res = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
    pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    team_map_lookup = {t["id"]: t["name"] for t in bootstrap_res["teams"]}
    
    player_static = {
        p["id"]: {
            "web_name": p["web_name"],
            "position": pos_map.get(p["element_type"]),
            "team": team_map_lookup.get(p["team"]),
            "value": p["now_cost"] / 10
        } 
        for p in bootstrap_res["elements"]
    }
    
    # 3) Match picks with predictions (Preserve all 15 players)
    team = []
    for pick in picks:
        elem = pick["element"]
        match = df_pred[df_pred["element"] == elem]
        
        pick_info = pick_map[elem]
        is_bench = pick_info["squad_position"] > 11
        multiplier = pick_info["multiplier"]

        if not match.empty:
            # Player HAS a fixture
            row = match.iloc[0]
            base_points = int(row["pred_points"])
            final_points = int(base_points * multiplier) 
            
            team.append({
                "element": int(elem),
                "name": str(row["name"]),
                "web_name": str(row.get("web_name", "")),
                "position": str(row["position"]),
                "is_bench": is_bench,
                "squad_order": pick_info["squad_position"],
                "value": float(row["value"]),
                "team": str(row["team_name"]),
                "opponent": str(row["opponent_name"]),
                "fdr": int(row["fixture_1_5_fdr"]),
                "form": float(row.get("form", 0.0)),
                "pred_points": final_points,
                "pred_points_base": base_points,
                "is_captain": pick_info["is_captain"],
                "is_vice_captain": pick_info["is_vice_captain"],
                "multiplier": multiplier,
                "news": str(row.get("news", "")),
            })
        else:
            # Player is BLANKING
            static = player_static.get(elem, {})
            team.append({
                "element": int(elem),
                "name": static.get("web_name", "Unknown"),
                "web_name": static.get("web_name", "Unknown"),
                "position": static.get("position", "N/A"),
                "is_bench": is_bench,
                "squad_order": pick_info["squad_position"],
                "value": static.get("value", 0.0),
                "team": static.get("team", "N/A"),
                "opponent": "-",
                "fdr": 5,
                "form": 0.0,
                "pred_points": 0,
                "pred_points_base": 0,
                "is_captain": pick_info["is_captain"],
                "is_vice_captain": pick_info["is_vice_captain"],
                "multiplier": multiplier,
                "news": "Blank Gameweek",
            })

    # 4) Total Point Calculation
    if active_chip == "bboost":
        total_pred = int(sum(p["pred_points"] for p in team))
    else:
        total_pred = int(sum(p["pred_points"] for p in team if not p["is_bench"]))

    return {
        "manager_id": manager_id,
        "manager_name": manager_name,
        "team_name": team_name,
        "season": "2025-26",
        "prediction_gw": target_gw,
        "team_source_gw": team_source_gw,
        "active_chip": str(active_chip) if active_chip else None,
        "available_chips": available_chips,
        "chip_history": chips_used,
        "team": team,
        "total_predicted_points": total_pred,
    }

@app.get("/manager/{manager_id}/planner")
def manager_planner(manager_id: int):
    """
    Planner endpoint:
    - Determines target GW automatically
    - Uses GW-1 picks as base
    - Uses GW predictions
    - Blocks creation if deadline passed (commented out)
    - NEVER auto-updates
    """

    target_gw = get_current_or_next_gw()
    free_transfers = calculate_free_transfers_live(manager_id, target_gw)
    bank = get_bank_live(manager_id, target_gw)
    
    source_gw = target_gw - 1

    if source_gw < 1:
        raise HTTPException(status_code=404, detail="No previous GW available")

    try:
        # Fetch previous GW picks ONLY (no fallback)
        r = requests.get(
            FPL_ENTRY_PICKS.format(manager_id=manager_id, gw=source_gw),
            timeout=10
        )
        r.raise_for_status()
        picks_data = r.json()
        picks = picks_data.get("picks", [])

        # Manager info
        r_manager = requests.get(
            f"https://fantasy.premierleague.com/api/entry/{manager_id}/",
            timeout=10
        )
        r_manager.raise_for_status()
        manager_info = r_manager.json()

        # Chip history
        r_history = requests.get(
            FPL_ENTRY_HISTORY.format(manager_id=manager_id),
            timeout=10
        )
        r_history.raise_for_status()
        history_data = r_history.json()
        raw_chips_used = history_data.get("chips", [])

        chips_used = [
            {"name": str(c["name"]), "gw": int(c["event"])}
            for c in raw_chips_used
        ]

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"FPL API error: {e}")

    if not picks:
        raise HTTPException(status_code=404, detail="No picks found")

    manager_name = f"{manager_info.get('player_first_name','')} {manager_info.get('player_last_name','')}".strip()
    team_name = manager_info.get("name", "")

    # CHIP RESET LOGIC
    all_chips_set = {"wildcard", "3xc", "bboost", "freehit"}

    if target_gw < 20:
        used_in_window = {str(c['name']) for c in chips_used if int(c['gw']) < 20}
    else:
        used_in_window = {str(c['name']) for c in chips_used if int(c['gw']) >= 20}

    available_chips = list(all_chips_set - used_in_window)

    # Build pick mapping (Note: Planner usually resets multipliers to 1/2 unless a chip is active)
    pick_map = {
        pick["element"]: {
            "is_captain": bool(pick.get("is_captain", False)),
            "is_vice_captain": bool(pick.get("is_vice_captain", False)),
            "multiplier": 2 if pick.get("is_captain", False) else 1,
            "squad_position": int(pick.get("position", 0)),
        }
        for pick in picks
    }

    # Run prediction for target GW
    df_pred = predict_next_gw_pipeline(
        pipeline_path="artifacts/fpl_xgb_pipeline.pkl",
        feature_list_path="artifacts/feature_list.json",
        df_full=df_2526,
        season="2025-26",
        gw_to_predict=target_gw,
        known_players=KNOWN_PLAYERS,
        player_id_col=PLAYER_ID_COL,
    )

    if df_pred is None or df_pred.empty:
        raise HTTPException(status_code=500, detail="Prediction failed")

    df_pred = enrich_predictions_with_bootstrap(df_pred, target_gw)
    df_pred = apply_availability_rule(df_pred)
    df_pred = apply_integer_rule(df_pred)
    enrichment_map = get_player_enrichment_map(target_gw)
    # --- ADDED: Fetch Bootstrap Data for "Blank" Player Lookups ---
    bootstrap_res = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
    pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    team_map_lookup = {t["id"]: t["name"] for t in bootstrap_res["teams"]}
    
    player_static = {
        p["id"]: {
            "web_name": p["web_name"],
            "position": pos_map.get(p["element_type"]),
            "team": team_map_lookup.get(p["team"]),
            "value": p["now_cost"] / 10
        } 
        for p in bootstrap_res["elements"]
    }

    team = []
    for pick in picks:
        elem = pick["element"]
        p_extra = enrichment_map.get(elem, {})
        match = df_pred[df_pred["element"] == elem]
        
        pick_info = pick_map[elem]
        is_bench = pick_info["squad_position"] > 11
        multiplier = pick_info["multiplier"]

        if not match.empty:
            # Player HAS a fixture
            row = match.iloc[0]
            base_points = int(row["pred_points"])
            final_points = int(base_points * multiplier)

            team.append({
                "element": int(elem),
                "name": str(row["name"]),
                "web_name": str(row.get("web_name", "")),
                "position": str(row["position"]),
                "is_bench": is_bench,
                "squad_order": pick_info["squad_position"],
                "value": float(row["value"]),
                "ownership_pct": float(row.get("ownership_pct", 0.0)),
                "team": str(row["team_name"]),
                "opponent": str(row["opponent_name"]),
                "fdr": int(row["fixture_1_5_fdr"]),
                "fixtures": row.get("fixtures", []),
                "next_3_fdrs": row.get("next_3_fdrs", []),
                "form": float(row.get("form", 0.0)),
                "pred_points": final_points,
                "pred_points_base": base_points,
                "is_captain": pick_info["is_captain"],
                "is_vice_captain": pick_info["is_vice_captain"],
                "multiplier": multiplier,
                "news": str(row.get("news", "")),
            })
        else:
            # Player is BLANKING - Keep in squad with 0 points
            static = player_static.get(elem, {})
            team.append({
                "element": int(elem),
                "name": static.get("web_name", "Unknown"),
                "web_name": static.get("web_name", "Unknown"),
                "position": player_static.get(elem, {}).get("position", "N/A"),
                "is_bench": is_bench,
                "squad_order": pick_info["squad_position"],
                "value": p_extra.get('now_cost', 0) / 10,
                "ownership_pct": p_extra.get('ownership_pct', 0.0),
                "team": player_static.get(elem, {}).get("team", "N/A"),
                "opponent": "-",
                "fdr": 0,
                "fixtures": p_extra.get('fixtures', []),
                "next_3_fdrs": p_extra.get('next_3_fdrs', []),
                "form": p_extra.get('form', 0.0),
                "pred_points": 0,
                "pred_points_base": 0,
                "is_captain": pick_info["is_captain"],
                "is_vice_captain": pick_info["is_vice_captain"],
                "multiplier": multiplier,
                "news": "Blank Gameweek",
            })

    total_pred = int(sum(p["pred_points"] for p in team if not p["is_bench"]))

    return {
        "manager_id": manager_id,
        "manager_name": manager_name,
        "team_name": team_name,
        "prediction_gw": target_gw,
        "team_source_gw": source_gw,
        "bank": bank,
        "free_transfers": free_transfers,
        "available_chips": available_chips,
        "chip_history": chips_used,
        "team": team,
        "total_predicted_points": total_pred,
    }

@app.get("/fixtures")
def fixtures():

    team_map = get_team_mapping()

    bootstrap = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
    events = bootstrap["events"]

    current_gw_obj = next((e for e in events if e["is_current"]), None)
    
    if current_gw_obj:
        # Check if the whole GW is finished. If not, stay on current.
        target_gw = current_gw_obj["id"]
    else:
        # Fallback to next GW if no current one is active
        target_gw = next((e["id"] for e in events if e["is_next"]), None)

    res = requests.get("https://fantasy.premierleague.com/api/fixtures/")
    fixtures = res.json()

    formatted = []

    for f in fixtures:

        dt = format_datetime(f.get("kickoff_time"))
        is_live = f.get("started") and not f.get("finished") and not f.get("finished_provisional")
        formatted.append({
            "gw": f["event"],
            "home": team_map[f["team_h"]],
            "away": team_map[f["team_a"]],
            "home_score": f["team_h_score"],
            "away_score": f["team_a_score"],
            "time": dt["time"],
            "display_date": dt["display_date"],
            "started": f["started"],
            "finished": f["finished"],
            "finished_provisional": f.get("finished_provisional"),
            "is_live": is_live
        })

    return {
        "target_gw": target_gw,
        "fixtures": formatted
    }