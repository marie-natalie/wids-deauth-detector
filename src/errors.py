import pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score

df = pd.read_csv("features.csv")
FEATURES = ["n_events", "dis_ratio", "n_flips", "mean_gap", "min_gap", "max_gap"]

print("Share of ALL attack windows with just 1 event:",
      round((df[df.label == 1]["n_events"] == 1).mean(), 3))

# same split as train.py
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
              .split(df, df["label"], groups=df["day"]))
test = df.iloc[te].copy()
test["pred"] = joblib.load("wids_rf.joblib").predict(test[FEATURES])

fn = test[(test.label == 1) & (test.pred == 0)]
fp = test[(test.label == 0) & (test.pred == 1)]
print("\n=== MISSED ATTACKS (false negatives):", len(fn))
print("events per missed window:\n", fn["n_events"].value_counts().head(6))
print("which devices:\n", fn["mac_address"].value_counts().head(5))
print("\n=== FALSE ALARMS (false positives):", len(fp))
print(fp[FEATURES].describe().round(2))
print("which devices:\n", fp["mac_address"].value_counts().head(5))

print("\n=== 5-fold cross-validation, split by DAY ===")
rows = []
for i, (a, b) in enumerate(GroupKFold(n_splits=5).split(df, df["label"], groups=df["day"])):
    m = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    m.fit(df.iloc[a][FEATURES], df.iloc[a]["label"])
    y, p = df.iloc[b]["label"], m.predict(df.iloc[b][FEATURES])
    rows.append(dict(fold=i, attack_windows=int(y.sum()), accuracy=accuracy_score(y, p),
                     precision=precision_score(y, p, zero_division=0),
                     recall=recall_score(y, p, zero_division=0)))
r = pd.DataFrame(rows).round(4)
print(r)
print(r[["accuracy", "precision", "recall"]].agg(["mean", "std"]).round(4))
