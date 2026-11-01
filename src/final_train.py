import pandas as pd, joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

df = pd.read_csv("features2.csv")
FEATURES = ["n_events", "dis_ratio", "n_flips", "mean_gap", "min_gap", "max_gap",
            "hist_60s", "hist_300s", "hist_flips_300s"]
y = df["label"].values
oof_rf = np.zeros(len(df), dtype=int)
oof_iso = np.zeros(len(df), dtype=int)

for a, b in GroupKFold(n_splits=5).split(df, y, groups=df["day"]):
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    rf.fit(df.iloc[a][FEATURES], y[a])
    oof_rf[b] = rf.predict(df.iloc[b][FEATURES])
    normal = df.iloc[a][y[a] == 0][FEATURES]                       # normal traffic only
    iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42).fit(normal)
    oof_iso[b] = (iso.predict(df.iloc[b][FEATURES]) == -1).astype(int)

for name, pred, fn in [("Random Forest", oof_rf, "cm_rf_final.png"),
                       ("Isolation Forest", oof_iso, "cm_iso_final.png")]:
    print(f"\n=== {name} (pooled out-of-fold, every window judged by a model that never saw its day) ===")
    print("Accuracy:", round(accuracy_score(y, pred), 4))
    print(confusion_matrix(y, pred))
    print(classification_report(y, pred, target_names=["normal", "attack"], digits=4))
    ConfusionMatrixDisplay(confusion_matrix(y, pred), display_labels=["normal", "attack"]).plot(values_format="d")
    plt.title(name + " - 5-fold by day"); plt.savefig(fn, dpi=200); plt.close()

# final models trained on everything, for the live/replay detector
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42).fit(df[FEATURES], y)
iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42).fit(df[y == 0][FEATURES])
joblib.dump(rf, "wids_rf_final.joblib"); joblib.dump(iso, "wids_iso_final.joblib")

imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
print("\nFeature importances:\n", imp.round(3).to_string())
imp.plot.barh(); plt.title("What the Random Forest learned"); plt.tight_layout()
plt.savefig("feature_importance.png", dpi=200)
print("\nSaved: cm_rf_final.png cm_iso_final.png feature_importance.png wids_*_final.joblib")
