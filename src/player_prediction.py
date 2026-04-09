import joblib
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

OUT = "outputs"

def train_player_model(X_train, X_test, y_train, y_test):
    print("\n📊 PLAYER RUNS MODEL COMPARISON")
    print("─" * 45)

    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)
    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5)

    models = {
        "Linear Regression": lr,
        "Random Forest": rf,
        "Gradient Boosting": gb
    }

    scores = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)

        scores[name] = (model, r2)

        print(f"  {name:25s}  R2={r2:.3f}  MAE={mae:.2f}")

    # Pick best
    best_name = max(scores, key=lambda k: scores[k][1])
    best_model, best_score = scores[best_name]

    print(f"\n  🏆 BEST: {best_name} → R2={best_score:.3f}")

    # Save model
    joblib.dump(best_model, f"{OUT}/player_model.pkl")
    joblib.dump({k: round(v*100, 2) for k, (_, v) in scores.items()},
            f"{OUT}/player_scores.pkl")

    return best_model, best_score