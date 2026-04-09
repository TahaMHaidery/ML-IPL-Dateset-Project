"""
toss_prediction.py  ─  Train all classifiers for toss outcome.
"""
import joblib
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier,
                               GradientBoostingClassifier, BaggingClassifier)
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV

OUT = "outputs"


def train_toss_model(X_train, X_test, y_train, y_test):
    print("\n📊 TOSS MODEL COMPARISON")
    print("─" * 50)

    knn = KNeighborsClassifier(n_neighbors=5)
    nb  = GaussianNB()
    dt  = DecisionTreeClassifier(max_depth=6, random_state=42)
    svm = SVC(kernel="rbf", probability=True, random_state=42)
    rf  = RandomForestClassifier(n_estimators=200, random_state=42)
    gb  = GradientBoostingClassifier(n_estimators=100, random_state=42)
    lr  = LogisticRegression(max_iter=5000, C=2.0, random_state=42)

    models = {
        "KNN":               knn,
        "Naive Bayes":       nb,
        "Decision Tree":     dt,
        "SVM":               svm,
        "Random Forest":     rf,
        "Gradient Boosting": gb,
        "Logistic Regression": lr,
    }

    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        scores[name] = (model, acc)
        print(f"  {name:25s}  {acc*100:.2f}%")

    # Voting
    voting = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("gb", gb)], voting="soft"
    )
    voting.fit(X_train, y_train)
    v_acc = accuracy_score(y_test, voting.predict(X_test))
    scores["Voting Ensemble"] = (voting, v_acc)
    print(f"  {'Voting Ensemble':25s}  {v_acc*100:.2f}%")

    # Bagging
    bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=6),
        n_estimators=100, random_state=42
    )
    bag.fit(X_train, y_train)
    b_acc = accuracy_score(y_test, bag.predict(X_test))
    scores["Bagging"] = (bag, b_acc)
    print(f"  {'Bagging':25s}  {b_acc*100:.2f}%")

    # GridSearch LR
    print("\n  🔍 GridSearch (Logistic Regression)...")
    gs = GridSearchCV(
        LogisticRegression(max_iter=5000, random_state=42),
        {"C": [0.1, 0.5, 1.0, 2.0, 5.0]},
        cv=5, scoring="accuracy", n_jobs=-1
    )
    gs.fit(X_train, y_train)
    gs_acc = accuracy_score(y_test, gs.best_estimator_.predict(X_test))
    scores["GridSearch LR"] = (gs.best_estimator_, gs_acc)
    print(f"  {'GridSearch LR':25s}  {gs_acc*100:.2f}%  C={gs.best_params_['C']}")

    best_name  = max(scores, key=lambda k: scores[k][1])
    best_model, best_acc = scores[best_name]
    print(f"\n  🏆 BEST: {best_name}  →  {best_acc*100:.2f}%")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_s = cross_val_score(best_model, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"  CV (5-fold): {cv_s.mean()*100:.2f}% ± {cv_s.std()*100:.2f}%")

    joblib.dump(best_model, f"{OUT}/toss_model.pkl")
    joblib.dump({k: round(v*100, 2) for k, (_, v) in scores.items()},
                f"{OUT}/toss_scores.pkl")

    return best_model, best_acc, scores