# src/py/build_dataset.py
from pathlib import Path
import pandas as pd
import numpy as np
import nfl_data_py as nfl

DATA = Path("data")
RAW = DATA / "raw"
FEAT = DATA / "features"
RAW.mkdir(parents=True, exist_ok=True)
FEAT.mkdir(parents=True, exist_ok=True)

# ---------- CONFIG ----------
SEASONS = list(range(2009, 2025))  # adjust as you like
ROLL_WINDOWS = [3, 4]              # last 3-4 games form
# Columns we’ll try to use from weekly data (they vary a bit by version)
CANDIDATE_COLS = {
    "points_for": ["points", "team_score", "score"],
    "points_against": ["opp_points", "opponent_score", "points_allowed"],
    "pass_yards_for": ["passing_yards", "pass_yards"],
    "rush_yards_for": ["rushing_yards", "rush_yards"],
    "turnovers_for": ["turnovers", "turnovers_total","turnovers_lost","giveaways"],
}
# If your weekly table has defensive allowed columns, you can add mappings here:
CANDIDATE_COLS.update({
    "pass_yards_against": ["opp_passing_yards","pass_yards_allowed"],
    "rush_yards_against": ["opp_rushing_yards","rush_yards_allowed"],
    "takeaways_for": ["takeaways"],  # optional
})

def pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def load_raw():
    print("Loading schedules...")
    sched = nfl.import_schedules(SEASONS)
    sched = sched.rename(columns={
        "home_team": "home",
        "away_team": "away",
        "home_score": "home_score",
        "away_score": "away_score",
    })
    # Keep core columns
    core_cols = [
        "game_id","season","week","gameday","home","away",
        "home_score","away_score","result","weekday","start_time"
    ]
    for c in core_cols:
        if c not in sched.columns:
            # tolerate missing, we'll handle later
            pass
    sched.to_parquet(RAW / "schedules.parquet", index=False)

    print("Loading sportsbook lines (SC lines)...")
    try:
        sc = nfl.import_sc_lines(SEASONS)  # may require update of nfl_data_py
    except Exception as e:
        print("import_sc_lines not available or failed:", e)
        sc = pd.DataFrame()

    if not sc.empty:
        # Typical fields include spread/total by book; we'll prefer consensus/close if present
        # Deduplicate to one row per game_id with preferred fields
        prefer_cols = [c for c in sc.columns if "spread" in c.lower() or "total" in c.lower()]
        keep = ["game_id","season","week","team"] + prefer_cols if "team" in sc.columns else ["game_id","season","week"] + prefer_cols
        sc = sc[[c for c in keep if c in sc.columns]].drop_duplicates("game_id")
        sc.to_parquet(RAW / "sc_lines.parquet", index=False)

    print("Loading weekly team data...")
    weekly = nfl.import_weekly_data(SEASONS)
    weekly.to_parquet(RAW / "weekly.parquet", index=False)

def normalize_weekly_team_col(weekly: pd.DataFrame) -> pd.DataFrame:
    # Map whatever the library provides → "team"
    team_like = None
    for cand in ["team", "recent_team", "team_abbr"]:
        if cand in weekly.columns:
            team_like = cand
            break
    if team_like is None:
        raise ValueError("weekly data is missing a team-like column (expected one of: team, recent_team, team_abbr)")
    weekly = weekly.rename(columns={team_like: "team"}).copy()
    # Make sure season/week are ints for consistent merges/rolling
    if "season" in weekly.columns:
        weekly["season"] = weekly["season"].astype(int)
    if "week" in weekly.columns:
        weekly["week"] = weekly["week"].astype(int)
    return weekly

def make_team_week_panel(sched: pd.DataFrame) -> pd.DataFrame:
    """
    Convert schedule into a team-week panel with one row per team per game.
    Useful when weekly() is missing some fields—we can at least get points for/against.
    """
    home = sched[["game_id","season","week","gameday","home","away","home_score","away_score"]].copy()
    home["team"] = home["home"]
    home["opponent"] = home["away"]
    home["points_for"] = home["home_score"]
    home["points_against"] = home["away_score"]

    away = sched[["game_id","season","week","gameday","home","away","home_score","away_score"]].copy()
    away["team"] = away["away"]
    away["opponent"] = away["home"]
    away["points_for"] = away["away_score"]
    away["points_against"] = away["home_score"]

    panel = pd.concat([home, away], ignore_index=True)
    panel = panel.drop(columns=["home","away"])
    return panel

def engineer_rolling_features(weekly: pd.DataFrame) -> pd.DataFrame:
    # Normalize team column and basic keys
    weekly = normalize_weekly_team_col(weekly)

    # Ensure required keys exist
    for col in ["season", "week", "team"]:
        if col not in weekly.columns:
            raise ValueError(f"weekly missing required column: {col}")

    df = weekly.copy()

    # Map source columns to canonical names (best-effort)
    for new_col, candidates in CANDIDATE_COLS.items():
        pick = pick_first_existing(df, candidates)
        df[new_col] = df[pick] if pick is not None else np.nan

    # Order for proper rolling (per team, per season, week ascending)
    df = df.sort_values(["team", "season", "week"]).reset_index(drop=True)
    group = df.groupby(["team", "season"], group_keys=False)

    # Ensure bases exist
    base_feats = [
        "points_for","points_against",
        "pass_yards_for","rush_yards_for",
        "pass_yards_against","rush_yards_against",
        "turnovers_for","takeaways_for"
    ]
    for f in base_feats:
        if f not in df.columns:
            df[f] = np.nan

    # Rolling means using ONLY prior games (shift(1))
    for w in ROLL_WINDOWS:
        for f in base_feats:
            df[f"{f}_r{w}"] = group[f].shift(1).rolling(w, min_periods=1).mean()
        df[f"to_diff_r{w}"] = df[f"takeaways_for_r{w}"] - df[f"turnovers_for_r{w}"]

    # Return keyed by season/week/team (no game_id dependency)
    keep_cols = ["season","week","team"] + [c for c in df.columns if any(c.endswith(f"_r{w}") for w in ROLL_WINDOWS)]
    return df[keep_cols].drop_duplicates(["season","week","team"])

def build():
    # read raw
    sched = pd.read_parquet(RAW / "schedules.parquet")
    weekly = pd.read_parquet(RAW / "weekly.parquet")
    sc_path = RAW / "sc_lines.parquet"
    sc = pd.read_parquet(sc_path) if sc_path.exists() else pd.DataFrame()

    # Targets on schedule
    if "home_score" in sched.columns and "away_score" in sched.columns:
        sched["home_win"] = (sched["home_score"] > sched["away_score"]).astype(int)
        sched["margin"] = sched["home_score"] - sched["away_score"]
    else:
        # future games will have NaNs
        sched["home_win"] = np.nan
        sched["margin"] = np.nan

    # Normalize team keys
    if "home" not in sched.columns and "home_team" in sched.columns:
        sched = sched.rename(columns={"home_team":"home","away_team":"away"})

    # --- Merge sportsbook lines defensively (handle dtype/column drift) ---
    def _to_str_key(df, col):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        return df

    def _pick_one(cols, *candidates):
        for c in candidates:
            hit = [x for x in cols if x.lower() == c.lower()]
            if hit:
                return hit[0]
        return None

    if not sc.empty:
        # Coerce game_id to string on BOTH sides to avoid object<int64> merge error
        sched = _to_str_key(sched, "game_id")
        sc    = _to_str_key(sc, "game_id")

        # Some nfl_data_py versions name columns slightly differently.
        # Try to find a reasonable spread/total column.
        sc_cols = list(sc.columns)
        spread_col = (
            _pick_one(sc_cols, "spread_close", "consensus_spread", "spread_line", "spread")
            or next((c for c in sc_cols if "spread" in c.lower()), None)
        )
        total_col = (
            _pick_one(sc_cols, "total_close", "consensus_total", "total_line", "total")
            or next((c for c in sc_cols if "total" in c.lower() or "over_under" in c.lower()), None)
        )

        # Reduce to one row per game_id with preferred columns
        keep_cols = ["game_id"]
        if spread_col: keep_cols.append(spread_col)
        if total_col:  keep_cols.append(total_col)
        lines = sc[keep_cols].drop_duplicates(subset=["game_id"]).copy()

        rename_map = {}
        if spread_col: rename_map[spread_col] = "spread_close"
        if total_col:  rename_map[total_col]  = "total_close"
        lines = lines.rename(columns=rename_map)

        # Final safe merge
        sched = sched.merge(lines, on="game_id", how="left")
    else:
        # Ensure columns exist even if no lines are available
        if "spread_close" not in sched.columns:
            sched["spread_close"] = np.nan
        if "total_close" not in sched.columns:
            sched["total_close"] = np.nan

    # Compute rolling form features from weekly team data
    rolling_team = engineer_rolling_features(weekly)

    # Build a home/away join: map team features onto each game row twice
    def attach_side_features(s: pd.DataFrame, side_col: str, prefix: str, rolling_team: pd.DataFrame):
    # s has schedule rows; we map the side team's rolling features by (season, week, team)
        side = s[["game_id","season","week",side_col]].rename(columns={side_col:"team"}).copy()
        merged = side.merge(rolling_team, on=["season","week","team"], how="left")
        feat_cols = [c for c in merged.columns if any(c.endswith(f"_r{w}") for w in ROLL_WINDOWS)]
        rename_map = {c: f"{prefix}_{c}" for c in feat_cols}
        return merged[["game_id"] + feat_cols].rename(columns=rename_map)

    home_feats = attach_side_features(sched, "home", "home", rolling_team)
    away_feats = attach_side_features(sched, "away", "away", rolling_team)

    games = sched.merge(home_feats, on="game_id", how="left").merge(away_feats, on="game_id", how="left")

    # Some useful flags
    def make_div_flag(df: pd.DataFrame) -> pd.Series:
        if "home_division" in df.columns and "away_division" in df.columns:
        # Compare as strings to be robust to categories/NaNs
            return (df["home_division"].astype(str) == df["away_division"].astype(str)).astype(float)
    # If division columns aren’t present, default to 0.0
        return pd.Series(0.0, index=df.index)

    games["is_divisional"] = make_div_flag(games)

    # Keep a clean modeling set
    keep = [
        "game_id","season","week","gameday","home","away",
        "home_score","away_score","home_win","margin",
        "spread_close","total_close","is_divisional"
    ] + [c for c in games.columns if c.startswith(("home_","away_")) and c.endswith(tuple([f"_r{w}" for w in ROLL_WINDOWS]))]

    model_df = games[[c for c in keep if c in games.columns]].copy()

    # Save
    # --- Robust save (Windows-safe) ---
    FEAT.mkdir(parents=True, exist_ok=True)

    out_parquet = (FEAT / "model_df.parquet").resolve()
    out_csv     = (FEAT / "model_df.csv").resolve()

    # Ensure pyarrow is present: pip install pyarrow
    try:
        # Use a file handle + explicit engine to avoid Windows path oddities
        with open(out_parquet, "wb") as f:
            model_df.to_parquet(f, index=False, engine="pyarrow")
        print(f"Saved Parquet → {out_parquet}")
    except Exception as e:
        print(f"[WARN] Parquet write failed: {e}")
        print("Falling back to CSV only.")
    finally:
        # Always write CSV for inspection
        model_df.to_csv(out_csv, index=False)
        print(f"Saved CSV → {out_csv}")
    print(f"Saved: {FEAT / 'model_df.parquet'} and .csv")

if __name__ == "__main__":
    load_raw()
    build()
