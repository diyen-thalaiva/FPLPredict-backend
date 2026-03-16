import requests
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"

_bootstrap_cache = None
_fixtures_cache = None

def get_bootstrap_data(force_refresh: bool = False) -> Dict:
    global _bootstrap_cache
    if _bootstrap_cache is not None and not force_refresh:
        return _bootstrap_cache
    
    try:
        r = requests.get(BOOTSTRAP_URL, timeout=10)
        r.raise_for_status()
        _bootstrap_cache = r.json()
        return _bootstrap_cache
    except Exception as e:
        logger.error(f"Bootstrap fetch failed: {e}")
        return {}

def get_fixtures_data(force_refresh: bool = False) -> List[Dict]:
    global _fixtures_cache
    if _fixtures_cache is not None and not force_refresh:
        return _fixtures_cache
    
    try:
        r = requests.get(FIXTURES_URL, timeout=10)
        r.raise_for_status()
        _fixtures_cache = r.json()
        return _fixtures_cache
    except Exception as e:
        logger.error(f"Fixtures fetch failed: {e}")
        return []

def get_player_enrichment_map(current_gw: int) -> Dict[int, dict]:
    bootstrap = get_bootstrap_data()
    fixtures = get_fixtures_data()
    
    # 1. Map Team IDs to Short Names (e.g., 1 -> "ARS")
    team_map = {t['id']: t['short_name'] for t in bootstrap.get('teams', [])}
    
    # 2. Build a Team Fixture Schedule
    # team_id -> { gw_number: {opp: "ARS", fdr: 2} }
    schedule = {}
    for f in fixtures:
        gw = f.get('event')
        if not gw: continue
        
        home_id = f['team_h']
        away_id = f['team_a']
        
        # Add home team info
        if home_id not in schedule: schedule[home_id] = {}
        schedule[home_id][gw] = {"opp": team_map.get(away_id), "fdr": f['team_h_difficulty']}
        
        # Add away team info
        if away_id not in schedule: schedule[away_id] = {}
        schedule[away_id][gw] = {"opp": team_map.get(home_id), "fdr": f['team_a_difficulty']}

    # 3. Create Player Enrichment Map
    enrichment_map = {}
    for player in bootstrap.get('elements', []):
        p_id = player['id']
        t_id = player['team']
        
        # Get fixtures for the next 3 Gameweeks starting from current_gw
        next_3_fixtures = []
        next_3_fdrs = []
        
        for i in range(3):
            target_gw = current_gw + i
            f_info = schedule.get(t_id, {}).get(target_gw)
            if f_info:
                next_3_fixtures.append(f_info['opp'])
                next_3_fdrs.append(f_info['fdr'])
            else:
                next_3_fixtures.append("-") # Blank Gameweek
                next_3_fdrs.append(2)
        
        transfers_in = player.get("transfers_in_event", 0)
        transfers_out = player.get("transfers_out_event", 0)

        total_transfers = transfers_in + transfers_out

        if total_transfers > 0:
            net_transfer_pct = round(((transfers_in - transfers_out) / total_transfers) * 100, 1)
        else:
            net_transfer_pct = 0.0


        enrichment_map[p_id] = {
            'web_name': player.get('web_name', ''),
            'news': player.get('news', ''),
            'now_cost': player.get('now_cost', 0), 
            'form': float(player.get('form', 0.0)),
            'ownership_pct': float(player.get('selected_by_percent', 0.0)),
            'total_points': player.get('total_points', 0), 
            'net_transfer_pct': net_transfer_pct,
            'fixtures': next_3_fixtures,
            'next_3_fdrs': next_3_fdrs
        }
    
    return enrichment_map

def enrich_predictions_with_bootstrap(df_pred, gw: int):
    """
    Enriches the dataframe with web_name, news, and the 3-match fixture/FDR data.
    """
    if 'element' not in df_pred.columns:
        return df_pred
    
    enrichment_map = get_player_enrichment_map(gw)
    
    df_pred['web_name'] = df_pred['element'].map(lambda x: enrichment_map.get(x, {}).get('web_name', ''))
    df_pred['news'] = df_pred['element'].map(lambda x: enrichment_map.get(x, {}).get('news', ''))
    df_pred['value'] = df_pred['element'].map(lambda x: enrichment_map.get(x, {}).get('now_cost', 0) / 10)
    df_pred['form'] = df_pred['element'].map(lambda x: enrichment_map.get(x, {}).get('form', 0.0))
    df_pred['ownership_pct'] = df_pred['element'].map(
        lambda x: enrichment_map.get(x, {}).get('ownership_pct', 0.0)
    )
    df_pred['total_points'] = df_pred['element'].map(
        lambda x: enrichment_map.get(x, {}).get('total_points', 0)
    )
    df_pred['net_transfer_pct'] = df_pred['element'].map(
        lambda x: enrichment_map.get(x, {}).get('net_transfer_pct', 0.0)
    )
    
    # Extract the lists into columns
    df_pred['fixtures'] = df_pred['element'].map(lambda x: enrichment_map.get(x, {}).get('fixtures', ["-", "-", "-"]))
    df_pred['next_3_fdrs'] = df_pred['element'].map(lambda x: enrichment_map.get(x, {}).get('next_3_fdrs', [2, 2, 2]))
    
    return df_pred