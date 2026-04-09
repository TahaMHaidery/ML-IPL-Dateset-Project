"""
app.py — IPL Predictor Dashboard (FINAL CLEAN VERSION)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ─── Load data ────────────────────────────────────────────────────
matches = pd.read_csv("data/matches.csv")
deliveries = pd.read_csv("data/deliveries.csv")

# ---------------- TEAM CLEANING ----------------
TEAM_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}

for col in ["team1", "team2", "toss_winner", "winner"]:
    if col in matches.columns:
        matches[col] = matches[col].replace(TEAM_MAP)

# ---------------- VENUE CLEANING (ADD THIS 🔥) ----------------

# Remove text after comma (very powerful fix)
matches["venue"] = matches["venue"].astype(str).str.replace(r",.*", "", regex=True)

# Remove extra spaces
matches["venue"] = matches["venue"].str.strip()

# Optional: normalize case
matches["venue"] = matches["venue"].str.title()

# ---------------- NOW CREATE UNIQUE VALUES ----------------

teams = sorted(set(matches["team1"].unique()) | set(matches["team2"].unique()))
venues = sorted(matches["venue"].dropna().unique())
players = sorted(deliveries["batter"].dropna().unique())

# ─── Load models ──────────────────────────────────────────────────
winner_model   = joblib.load("outputs/winner_model.pkl")
winner_columns = joblib.load("outputs/winner_columns.pkl")
winner_scaler  = joblib.load("outputs/winner_scaler.pkl")
team_stats     = joblib.load("outputs/team_stats.pkl")
h2h_stats      = joblib.load("outputs/h2h_stats.pkl")
venue_stats    = joblib.load("outputs/venue_stats.pkl")
form_stats     = joblib.load("outputs/form_stats.pkl")
winner_scores  = joblib.load("outputs/winner_scores.pkl")

toss_model     = joblib.load("outputs/toss_model.pkl")
toss_columns   = joblib.load("outputs/toss_columns.pkl")
toss_scaler    = joblib.load("outputs/toss_scaler.pkl")
toss_team_stats= joblib.load("outputs/toss_team_stats.pkl")
toss_h2h       = joblib.load("outputs/toss_h2h_stats.pkl")
toss_scores    = joblib.load("outputs/toss_scores.pkl")

score_model    = joblib.load("outputs/score_model.pkl")
score_scaler   = joblib.load("outputs/score_scaler.pkl")
score_scores   = joblib.load("outputs/score_scores.pkl")
score_columns   = joblib.load("outputs/score_columns.pkl")

player_model   = joblib.load("outputs/player_model.pkl")
player_cols    = joblib.load("outputs/player_columns.pkl")
player_scaler  = joblib.load("outputs/player_scaler.pkl")
player_scores   = joblib.load("outputs/player_scores.pkl")


# ─── Helper Functions ─────────────────────────────────────────────

def safe_le(le, val):
    if val in le.classes_:
        return le.transform([val])[0]
    return 0


def get_best_model(model_dict):
    best_model = None
    best_score = -1

    for model, score in model_dict.items():
        if score > best_score:
            best_score = score
            best_model = model

    return {
        "model": best_model,
        "score": round(best_score, 2)
    }


def get_best_score_model(score_dict):

    # If single value
    if isinstance(score_dict, (int, float)):
        return {
            "model": "Score Model",
            "r2": round(score_dict * 100, 2),
            "rmse": 0
        }

    # If dictionary
    if isinstance(score_dict, dict):

        best_model = None
        best_r2 = -1
        best_rmse = 0

        for model, metrics in score_dict.items():
            if not isinstance(metrics, dict):
                continue

            r2 = metrics.get("r2", 0)

            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_rmse = metrics.get("rmse", 0)

        return {
            "model": best_model or "N/A",
            "r2": round(best_r2 * 100, 2),
            "rmse": round(best_rmse, 2)
        }

    return {
        "model": "N/A",
        "r2": 0,
        "rmse": 0
    }


# ─── Page Routes ──────────────────────────────────────────────────

@app.route("/")
def dashboard():
    return render_template("dashboard.html", teams=teams, venues=venues)

@app.route("/winner")
def winner_page():
    return render_template("winner.html", teams=teams, venues=venues)

@app.route("/toss")
def toss_page():
    return render_template("toss.html", teams=teams)

@app.route("/score")
def score_page():
    return render_template("score.html", teams=teams, venues=venues)

@app.route("/h2h")
def h2h_page():
    return render_template("h2h.html", teams=teams)

@app.route("/player")
def player_page():
    return render_template("player.html", teams=teams, players=players)

# ─── Prediction APIs ──────────────────────────────────────────────

@app.route("/predict_winner", methods=["POST"])
def predict_winner():
    d = request.json

    t1 = d["team1"]; t2 = d["team2"]
    tw = d["toss_winner"]; td = d["toss_decision"]
    venue = d["venue"]

    row = {
        "team1_win_rate": team_stats.get(t1, 50),
        "team2_win_rate": team_stats.get(t2, 50),
        "h2h_t1_vs_t2": h2h_stats.get((t1, t2), 50),
        "t1_venue_wr": venue_stats.get((t1, venue), 50),
        "t2_venue_wr": venue_stats.get((t2, venue), 50),
        "t1_form": form_stats.get(t1, 50),
        "t2_form": form_stats.get(t2, 50),
        "toss_winner_is_t1": 1 if tw == t1 else 0,
        "toss_decision_bat": 1 if td == "bat" else 0,
        f"team1_{t1}": 1,
        f"team2_{t2}": 1,
        f"venue_{venue}": 1,
    }

    X = pd.DataFrame([row]).reindex(columns=winner_columns, fill_value=0)
    X = winner_scaler.transform(X)

    pred = winner_model.predict(X)[0]
    winner = t1 if pred == 1 else t2

    return jsonify({"winner": winner})


@app.route("/predict_toss", methods=["POST"])
def predict_toss():
    d = request.json
    t1 = d["team1"]; t2 = d["team2"]

    row = {
        "t1_win_rate": toss_team_stats.get(t1, 50),
        "t2_win_rate": toss_team_stats.get(t2, 50),
        "h2h": toss_h2h.get((t1, t2), 50),
        f"team1_{t1}": 1,
        f"team2_{t2}": 1,
    }

    X = pd.DataFrame([row]).reindex(columns=toss_columns, fill_value=0)
    X = toss_scaler.transform(X)

    pred = toss_model.predict(X)[0]
    toss_winner = t1 if pred == 1 else t2

    return jsonify({"toss_winner": toss_winner})


@app.route("/predict_score", methods=["POST"])
def predict_score():
    d = request.json

    over = float(d["over"])
    runs = float(d.get("cum_runs", over * 8))
    wickets = float(d.get("cum_wickets", 0))

    # ✅ REQUIRED FEATURES (FIX)
    crr = runs / over if over > 0 else 0

    venue = d.get("venue", "")
    venue_bat_avg = venue_stats.get(venue, 160)

    # ✅ CREATE INPUT
    X = pd.DataFrame([{
        "over": over,
        "cum_runs": runs,
        "cum_wickets": wickets,
        "runs_last5": runs,
        "crr": crr,                   # ✅ FIX
        "venue_bat_avg": venue_bat_avg,  # ✅ FIX
        "bat_enc": 0,
        "bowl_enc": 0,
        "venue_enc": 0,
    }])

    # ✅ FIX ORDER + MISSING COLS (CRITICAL)
    X = X.reindex(columns=score_columns, fill_value=0)

    X = score_scaler.transform(X)
    pred = score_model.predict(X)[0]

    return jsonify({
        "score": int(pred),
        "low": int(pred - 10),
        "high": int(pred + 10)
    })


# ─── Dashboard API ────────────────────────────────────────────────

@app.route("/api/stats")
def api_stats():

    wins = matches["winner"].value_counts().head(10).to_dict()

    toss_impact = matches.dropna(subset=["winner", "toss_winner"])
    toss_win_pct = round(
        (toss_impact["toss_winner"] == toss_impact["winner"]).mean() * 100, 1
    )

    toss_dec = matches["toss_decision"].value_counts().to_dict()
    # season_counts = matches["season"].value_counts().sort_index().to_dict()
    venue_counts = matches["venue"].value_counts().head(8).to_dict()

    # Convert season to string first
    matches["season"] = matches["season"].astype(str)

    # Extract numeric year for sorting
    matches["season_num"] = matches["season"].str.extract(r'(\d{4})').astype(int)

    # Count matches
    season_counts_df = matches.groupby("season_num").size().reset_index(name="matches")

    # Sort properly
    season_counts_df = season_counts_df.sort_values("season_num")

    # Convert to dict
    season_counts = {
        str(row["season_num"]): int(row["matches"])
        for _, row in season_counts_df.iterrows()
    }

    best_winner = get_best_model(winner_scores)
    best_toss = get_best_model(toss_scores)
    best_score = get_best_score_model(score_scores)
    best_player = get_best_model(player_scores)

    return jsonify({
        "total_matches": int(len(matches)),
        "total_seasons": int(matches["season"].nunique()),
        "total_teams": int(len(teams)),
        "toss_win_pct": toss_win_pct,

        "top_teams": wins,
        "toss_decisions": toss_dec,
        "season_matches": {str(k): v for k, v in season_counts.items()},
        "top_venues": venue_counts,

        "best_models": {
            "winner": best_winner,
            "toss": best_toss,
            "score": best_score,
            "player" : best_player
        }
    })


@app.route("/api/head2head")
def head2head():
    t1 = request.args.get("t1", "")
    t2 = request.args.get("t2", "")

    if not t1 or not t2:
        return jsonify({})

    m = matches[((matches["team1"] == t1) & (matches["team2"] == t2)) |
                ((matches["team1"] == t2) & (matches["team2"] == t1))]

    return jsonify({
        "total": len(m),
        "t1_wins": int((m["winner"] == t1).sum()),
        "t2_wins": int((m["winner"] == t2).sum()),
    })

@app.route("/predict_player", methods=["POST"])
def predict_player():
    data = request.get_json()

    import pandas as pd

    input_data = pd.DataFrame([{
        "batting_team": data.get("batting_team"),
        "bowling_team": data.get("bowling_team"),
        "player_avg": float(data.get("player_avg", 0)),
        "last5_avg": float(data.get("last5_avg", 0)),
        "strike_rate": 120
    }])

    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=player_cols, fill_value=0)
    input_scaled = player_scaler.transform(input_data)

    pred = player_model.predict(input_scaled)[0]

    return jsonify({"prediction": round(pred, 2)})


# ─── Run ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🏏 IPL Dashboard running → http://localhost:5000")
    app.run(debug=True)