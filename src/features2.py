import pandas as pd
f = pd.read_csv("features.csv", parse_dates=["win"])
parts = []
for dev, g in f.groupby("mac_address"):
    g = g.sort_values("win").set_index("win")
    g["hist_60s"] = g["n_events"].rolling("60s", closed="left").sum()
    g["hist_300s"] = g["n_events"].rolling("300s", closed="left").sum()
    g["hist_flips_300s"] = g["n_flips"].rolling("300s", closed="left").sum()
    parts.append(g.reset_index())
out = pd.concat(parts).fillna(0).sort_values("win").reset_index(drop=True)
out.to_csv("features2.csv", index=False)
print(out.shape)
print(out.groupby("label")[["hist_60s", "hist_300s", "hist_flips_300s"]].median())
