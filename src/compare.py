import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score

df = pd.read_csv("features2.csv")
BASE = ["n_events", "dis_ratio", "n_flips", "mean_gap", "min_gap", "max_gap"]
HIST = BASE + ["hist_60s", "hist_300s", "hist_flips_300s"]

def cv(features, name):
    rows, lone_rec = [], []
    for a, b in GroupKFold(n_splits=5).split(df, df["label"], groups=df["day"]):
        m = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        m.fit(df.iloc[a][features], df.iloc[a]["label"])
        t = df.iloc[b]; p = m.predict(t[features])
        rows.append([accuracy_score(t.label, p), precision_score(t.label, p, zero_division=0),
                     recall_score(t.label, p, zero_division=0)])
        lone = (t.label == 1) & (t.n_events == 1)
        lone_rec.append((p[lone.values] == 1).mean())
    r = pd.DataFrame(rows, columns=["accuracy", "precision", "recall"])
    print(f"\n=== {name} ===")
    print(r.agg(["mean", "std"]).round(4))
    print("Recall on lone-event attack windows (mean over folds):", round(sum(lone_rec) / 5, 3))

cv(BASE, "BASELINE (no history)")
cv(HIST, "WITH HISTORY")
