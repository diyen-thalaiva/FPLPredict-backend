# inference_core.py
import os
import json
from pathlib import Path

import pandas as pd
import joblib

# ------------------------------------------------------------------
# Default artifact locations (you can change these in your backend)
# ------------------------------------------------------------------
DEFAULT_ARTIFACT_DIR = Path("artifacts")
DEFAULT_PIPELINE_PATH = DEFAULT_ARTIFACT_DIR / "fpl_xgb_pipeline.pkl"
DEFAULT_FEATURE_LIST_PATH = DEFAULT_ARTIFACT_DIR / "feature_list.json"
DEFAULT_DF_PATH = DEFAULT_ARTIFACT_DIR / "df_all.parquet"


# ------------------------------------------------------------------
# 1. Core helper: prepare_for_inference
# ------------------------------------------------------------------
def prepare_for_inference(
    df_full,                    # full dataframe with rolling features
    season,                     # target season string, e.g. "2025-26"
    gw_to_predict,              # integer GW to predict, e.g. 1..38
    feature_cols_infer,         # list of features used by model (REQUIRED here)
    use_crossseason=False,      # if True, prefer *_cross_last3/_cross_last5 columns if available
    apply_coldstart_impute=True,# whether to apply cold-start imputation for new players
    known_players=None,         # optional set/list of known player IDs (element or name)
    player_id_col=None          # optional (defaults to 'element' if present else 'name')
):
    """
    Returns: df_inf (rows for season/GW), X_inf (DataFrame ready for model.predict)

    Notes:
      - df_full must already contain rolling features (shift+rolling) computed.
      - Cold-start imputation: zero scoring rolling features, minutes fallback to team-position average.
      - feature_cols_infer must be passed in this module (e.g. expected_features from feature_list.json).
    """
    if feature_cols_infer is None:
        raise ValueError(
            "feature_cols_infer must be provided to prepare_for_inference(). "
            "Pass expected_features from the saved feature list."
        )

    if player_id_col is None:
        player_id_col = "element" if "element" in df_full.columns else "name"

    # Filter rows for inference (season & GW)
    df_inf = df_full[(df_full["season"] == season) & (df_full["GW"] == gw_to_predict)].copy()
    if df_inf.empty:
        print(f"[prepare_for_inference] No rows found for season={season} GW={gw_to_predict}")
        return df_inf, None

    # Compute known_players set if not given
    if known_players is None:
        known_players = set(df_full[player_id_col].dropna().unique())

    # Flag new players (by ID column: element OR name)
    df_inf["is_new_player"] = ~df_inf[player_id_col].isin(known_players)

    # Optional cross-season mapping (if you ever add *_cross_last3/_cross_last5 features)
    if use_crossseason:
        for src_col in list(df_inf.columns):
            if "_cross_last3" in src_col:
                target = src_col.replace("_cross_last3", "_last3")
                if target not in df_inf.columns:
                    df_inf[target] = df_inf[src_col]
            if "_cross_last5" in src_col:
                target = src_col.replace("_cross_last5", "_last5")
                if target not in df_inf.columns:
                    df_inf[target] = df_inf[src_col]

    # Cold-start imputation for new players
    if apply_coldstart_impute:
        # features to zero for new players
        zero_cols = [c for c in [
            "total_points_last3","total_points_last5",
            "goals_scored_last3","assists_last3","ict_index_last3",
            "team_goals_last3","team_points_last3",
            "opp_goals_last3","opp_points_last3"
        ] if c in df_inf.columns]

        # set them to 0.0 for new players
        if zero_cols:
            df_inf.loc[df_inf["is_new_player"], zero_cols] = 0.0

        # minutes fallback: team-position average from df_full
        if ("team_name" in df_full.columns) and ("position_ord" in df_full.columns) and ("minutes" in df_full.columns):
            teampos_avg = df_full.groupby(["team_name","position_ord"])["minutes"].mean().to_dict()

            def _teampos_minutes(row):
                return teampos_avg.get(
                    (row.get("team_name"), row.get("position_ord")),
                    df_full["minutes"].median()
                )

            if "minutes_last3" in df_inf.columns:
                df_inf.loc[df_inf["is_new_player"], "minutes_last3"] = (
                    df_inf.loc[df_inf["is_new_player"]].apply(_teampos_minutes, axis=1)
                )
            if "minutes_last5" in df_inf.columns:
                df_inf.loc[df_inf["is_new_player"], "minutes_last5"] = (
                    df_inf.loc[df_inf["is_new_player"]].apply(_teampos_minutes, axis=1)
                )

        # If some feature cols are still NaN, fill with median of df_full (safe fallback)
        for f in feature_cols_infer:
            if f in df_inf.columns and df_inf[f].isnull().any():
                med = df_full[f].median() if f in df_full.columns else 0.0
                df_inf[f] = df_inf[f].fillna(med)

    # Build X for inference: ensure all feature cols exist; if missing, create and fill with median
    X_inf = pd.DataFrame(index=df_inf.index)
    for f in feature_cols_infer:
        if f in df_inf.columns:
            X_inf[f] = df_inf[f].astype(float)
        else:
            med = df_full[f].median() if f in df_full.columns else 0.0
            X_inf[f] = med

    # final safety: replace any remaining NaNs with 0
    X_inf = X_inf.fillna(0.0)

    return df_inf, X_inf


# ------------------------------------------------------------------
# 2. Load pipeline + feature list
# ------------------------------------------------------------------
def _load_pipeline_and_features(
    pipeline_path=DEFAULT_PIPELINE_PATH,
    feature_list_path=DEFAULT_FEATURE_LIST_PATH,
):
    """Load saved pipeline and expected feature list (raises helpful errors if missing)."""
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Pipeline file not found at: {pipeline_path}")
    pipe = joblib.load(pipeline_path)

    if not os.path.exists(feature_list_path):
        # best-effort fallback: try to extract from an XGBoost booster inside the pipeline
        if hasattr(pipe, "named_steps"):
            for step_name, step in pipe.named_steps.items():
                if hasattr(step, "get_booster"):
                    try:
                        booster = step.get_booster()
                        feat_names = booster.feature_names
                        return pipe, list(feat_names)
                    except Exception:
                        pass
        raise FileNotFoundError(f"Feature list JSON not found at: {feature_list_path}")

    with open(feature_list_path, "r") as fh:
        expected_features = json.load(fh)
    return pipe, expected_features


# ------------------------------------------------------------------
# 3. Align X_inf to expected feature order
# ------------------------------------------------------------------
def _align_and_validate_X(X_inf, df_full, expected_features, use_crossseason=False):
    """
    Ensure X_inf contains expected_features in same order.
    If some expected features missing, fill with median from df_full or 0.0.
    If use_crossseason=True and X_inf contains *_cross_last* columns, copy them into *_last* names.
    """
    X = X_inf.copy()

    # optional: cross-season mapping on X as well
    if use_crossseason:
        for col in list(X.columns):
            if "_cross_last3" in col:
                target = col.replace("_cross_last3", "_last3")
                if target not in X.columns:
                    X[target] = X[col]
            if "_cross_last5" in col:
                target = col.replace("_cross_last5", "_last5")
                if target not in X.columns:
                    X[target] = X[col]

    # Fill missing expected features with median from df_full or zero
    for f in expected_features:
        if f not in X.columns:
            if (df_full is not None) and (f in df_full.columns):
                X[f] = df_full[f].median()
            else:
                X[f] = 0.0

    # Reorder columns to expected order
    X = X[expected_features].copy()

    # Final safety fill
    X = X.fillna(0.0)

    return X


# ------------------------------------------------------------------
# 4. Main inference wrapper
# ------------------------------------------------------------------
def predict_next_gw_pipeline(
    pipeline_path=DEFAULT_PIPELINE_PATH,
    feature_list_path=DEFAULT_FEATURE_LIST_PATH,
    df_full=None,                # full historical dataframe (must include rolling columns)
    season=None,                 # e.g. "2024-25"
    gw_to_predict=None,          # target GW index - inference will use rows with GW == gw_to_predict
    use_crossseason=False,       # if you wish to map cross-season columns into expected names (inference-only)
    apply_coldstart_impute=True, # delegates to prepare_for_inference for cold-start logic
    known_players=None,          # optional set/list of known players to detect new players
    player_id_col=None           # 'element' or 'name'
):
    """
    Full pipeline-safe inference wrapper:
      - loads saved pipeline & expected feature order
      - calls prepare_for_inference(...) to get df_inf, X_inf
      - aligns X_inf -> expected features
      - runs pipeline.predict(X_aligned)
      - attaches preds and diagnostics to df_inf

    Returns: df_out (DataFrame)
    """
    if df_full is None:
        raise ValueError("You must pass df_full (the full dataframe with rolling features).")
    if season is None or gw_to_predict is None:
        raise ValueError("You must pass both season and gw_to_predict.")

    # 1) load pipeline & expected features
    pipe, expected_features = _load_pipeline_and_features(
        pipeline_path=pipeline_path,
        feature_list_path=feature_list_path,
    )

    # 2) choose player ID column
    if player_id_col is None:
        player_id_col = "element" if "element" in df_full.columns else "name"

    # 3) prepare inference rows + raw feature matrix
    df_inf, X_inf = prepare_for_inference(
        df_full=df_full,
        season=season,
        gw_to_predict=gw_to_predict,
        feature_cols_infer=expected_features,   # use the same features the model expects
        use_crossseason=use_crossseason,
        apply_coldstart_impute=apply_coldstart_impute,
        known_players=known_players,
        player_id_col=player_id_col,
    )

    if X_inf is None or X_inf.shape[0] == 0:
        # nothing to predict
        return df_inf

    # 4) Align X_inf to expected features (extra safety)
    X_aligned = _align_and_validate_X(
        X_inf,
        df_full=df_full,
        expected_features=expected_features,
        use_crossseason=use_crossseason,
    )

    # 5) Predict using the loaded pipeline (which contains preprocessing + estimator)
    preds = pipe.predict(X_aligned)

    # 6) Attach predictions and diagnostics to df_inf
    df_out = df_inf.copy()
    df_out["pred_points"] = preds

    # Ensure is_new_player exists (in case prepare_for_inference is changed in future)
    if "is_new_player" not in df_out.columns:
        if known_players is not None:
            known = set(known_players)
        else:
            known = set(df_full[player_id_col].dropna().unique())
        df_out["is_new_player"] = ~df_out[player_id_col].isin(known)

    # career_rows + confidence proxy (simple)
    if "element" in df_out.columns and "element" in df_full.columns:
        counts = df_full.groupby("element").size().to_dict()
        df_out["career_rows"] = df_out["element"].map(lambda x: counts.get(x, 0))
    else:
        df_out["career_rows"] = 0
    df_out["confidence_proxy"] = (df_out["career_rows"] / 5.0).clip(0, 1)

    df_out["inference_gw"] = gw_to_predict
    df_out["inference_season"] = season

    # safer sorting
    if "team_name" in df_out.columns:
        df_out = df_out.sort_values(["team_name", "pred_points"],
                                    ascending=[True, False])
    else:
        df_out = df_out.sort_values("pred_points", ascending=False)

    return df_out


# ------------------------------------------------------------------
# 5. Convenience: load df_all + model + feature list in one shot
# ------------------------------------------------------------------
def load_bundle(
    df_path: Path = DEFAULT_DF_PATH,
    pipeline_path: Path = DEFAULT_PIPELINE_PATH,
    feature_list_path: Path = DEFAULT_FEATURE_LIST_PATH,
):
    """Load df_all + pipeline + expected feature list for backend startup."""
    if df_path.suffix == ".parquet":
        df_all = pd.read_parquet(df_path)
    else:
        df_all = pd.read_csv(df_path)

    pipe, expected_features = _load_pipeline_and_features(
        pipeline_path=str(pipeline_path),
        feature_list_path=str(feature_list_path),
    )

    return df_all, pipe, expected_features
