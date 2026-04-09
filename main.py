"""
main.py  —  Run full training pipeline.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing     import preprocess_winner_data, preprocess_toss_data, preprocess_score_data, preprocess_player_data
from src.winner_prediction import train_winner_model
from src.toss_prediction   import train_toss_model
from src.score_prediction  import train_score_model
from src.player_prediction import train_player_model
from src.eda import perform_eda

def main():
    print("=" * 55)
    print("  🏏  IPL ML PROJECT — TRAINING PIPELINE")
    print("=" * 55)

    print("\n[1/5] Winner Prediction …")
    X_tr, X_te, y_tr, y_te = preprocess_winner_data()
    train_winner_model(X_tr, X_te, y_tr, y_te)

    print("\n[2/5] Toss Prediction …")
    X_tr, X_te, y_tr, y_te = preprocess_toss_data()
    train_toss_model(X_tr, X_te, y_tr, y_te)

    print("\n[3/5] Score Prediction …")
    X_tr, X_te, y_tr, y_te = preprocess_score_data()
    train_score_model(X_tr, X_te, y_tr, y_te)

    print("\n[4/5] Player Runs Prediction …")
    X_tr, X_te, y_tr, y_te = preprocess_player_data()
    train_player_model(X_tr, X_te, y_tr, y_te)

    print("\n[4/5] Player Runs Prediction …")
    perform_eda()

    print("\n" + "=" * 55)
    print("  ✅  ALL MODELS TRAINED & SAVED TO outputs/")
    print("=" * 55)

if __name__ == "__main__":
    main()