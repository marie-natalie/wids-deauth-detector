import pandas as pd, joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

df = pd.read_csv("features.csv")
FEATURES = ["n_events", "dis_ratio", "n_flips", "mean_gap", "min_gap", "max_gap"]

gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
tr, te = next(gss.split(df, df["label"], groups=df["day"]))
train, test = df.iloc[tr], df.iloc[te]
print("Train:", train["label"].value_counts().to_dict(), "| Test:", test["label"].value_counts().to_dict())

# ---- Model 1: Random Forest (supervised) ----
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf.fit(train[FEATURES], train["label"])
p = rf.predict(test[FEATURES])
print("\n=== RANDOM FOREST ===\nAccuracy:", round(accuracy_score(test["label"], p), 4))
print(classification_report(test["label"], p, target_names=["normal", "attack"], digits=4))
print("Feature importance:", dict(zip(FEATURES, rf.feature_importances_.round(3))))
ConfusionMatrixDisplay(confusion_matrix(test["label"], p), display_labels=["normal", "attack"]).plot()
plt.title("Random Forest"); plt.savefig("cm_random_forest.png", dpi=200); plt.close()
joblib.dump(rf, "wids_rf.joblib")

# ---- Model 2: Isolation Forest (unsupervised, trained on NORMAL only) ----
normal_only = train[train["label"] == 0][FEATURES]
iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42).fit(normal_only)
q = (iso.predict(test[FEATURES]) == -1).astype(int)     # -1 = anomaly -> attack
print("\n=== ISOLATION FOREST (never saw an attack) ===\nAccuracy:", round(accuracy_score(test["label"], q), 4))
print(classification_report(test["label"], q, target_names=["normal", "attack"], digits=4))
ConfusionMatrixDisplay(confusion_matrix(test["label"], q), display_labels=["normal", "attack"]).plot()
plt.title("Isolation Forest"); plt.savefig("cm_isolation_forest.png", dpi=200); plt.close()
joblib.dump(iso, "wids_iso.joblib")
