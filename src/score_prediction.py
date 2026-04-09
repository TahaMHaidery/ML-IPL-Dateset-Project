"""
score_prediction.py  —  Train & compare regressors for first-innings score.
"""

import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV

OUT = "outputs"


def train_score_model(X_train, X_test, y_train, y_test):
    print("\n📊 SCORE MODEL COMPARISON")
    print("─" * 55)

    lr  = LinearRegression()
    rid = Ridge(alpha=1.0)
    dt  = DecisionTreeRegressor(max_depth=10, random_state=42)
    rf  = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    gb  = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                     max_depth=6, random_state=42)

    models = {
        "Linear Regression":    lr,
        "Ridge":                rid,
        "Decision Tree":        dt,
        "Random Forest":        rf,
        "Gradient Boosting":    gb,
    }

    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mse  = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, pred)
        mae  = mean_absolute_error(y_test, pred)
        scores[name] = (model, rmse, r2, mae)
        print(f"  {name:25s}  RMSE={rmse:.2f}  R²={r2:.4f}  MAE={mae:.2f}")

    # GridSearch on RF
    print("\n  🔍 GridSearch (RF)...")
    gs = GridSearchCV(
        RandomForestRegressor(random_state=42),
        {"n_estimators": [200, 300], "max_depth": [12, 15, None]},
        cv=3, scoring="r2", n_jobs=-1
    )
    gs.fit(X_train, y_train)
    pred = gs.best_estimator_.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2   = r2_score(y_test, pred)
    mae  = mean_absolute_error(y_test, pred)
    scores["GridSearch RF"] = (gs.best_estimator_, rmse, r2, mae)
    print(f"  {'GridSearch RF':25s}  RMSE={rmse:.2f}  R²={r2:.4f}  params={gs.best_params_}")

    # Pick best by RMSE (lower = better)
    best_name  = min(scores, key=lambda k: scores[k][1])
    best_model = scores[best_name][0]
    best_rmse  = scores[best_name][1]
    best_r2    = scores[best_name][2]

    print(f"\n  🏆 BEST: {best_name}  →  RMSE={best_rmse:.2f}  R²={best_r2:.4f}")

    cv = cross_val_score(best_model, X_train, y_train, cv=5, scoring="r2")
    print(f"  CV R² (5-fold): {cv.mean():.4f} ± {cv.std():.4f}")

    joblib.dump(best_model, f"{OUT}/score_model.pkl")
    joblib.dump(
        {k: {"rmse": round(v[1], 2), "r2": round(v[2], 4)} for k, v in scores.items()},
        f"{OUT}/score_scores.pkl"
    )
    
    return best_model, best_rmse, scores