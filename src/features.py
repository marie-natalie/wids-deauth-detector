import pandas as pd

F = "Wi-Fi-Association_Disassociation-Dataset/Aassociation-Disassociation_Dataset.csv"
df = pd.read_csv(F)
df["ts"] = pd.to_datetime(df["@timestamp"], format="%b %d, %Y @ %H:%M:%S.%f")
df = df.sort_values(["mac_address", "ts"]).reset_index(drop=True)

df["is_dis"] = (df["association"] == "disassociated").astype(int)
df["is_att"] = (df["dataType"] == "attack").astype(int)
# seconds since this device's previous event (capped; first event gets the cap)
df["gap"] = df.groupby("mac_address")["ts"].diff().dt.total_seconds().fillna(3600).clip(upper=3600)
# 1 when the device switched state (associated <-> disassociated)
df["flip"] = (df.groupby("mac_address")["is_dis"].diff().abs() == 1).astype(int)
df["win"] = df["ts"].dt.floor("10s")

f = df.groupby(["mac_address", "win"]).agg(
    n_events=("is_dis", "size"),
    n_dis=("is_dis", "sum"),
    n_flips=("flip", "sum"),
    mean_gap=("gap", "mean"),
    min_gap=("gap", "min"),
    max_gap=("gap", "max"),
    n_att=("is_att", "sum"),
).reset_index()

f["dis_ratio"] = f["n_dis"] / f["n_events"]
f["label"] = (f["n_att"] > 0).astype(int)          # 1 = attack window
f["day"] = f["win"].dt.strftime("%Y-%m-%d")        # used only for splitting

mixed = ((f["n_att"] > 0) & (f["n_att"] < f["n_events"])).sum()
print("Windows:", len(f), "| attack:", f["label"].sum(), "| normal:", (f["label"] == 0).sum())
print("Windows mixing normal+attack events:", mixed)
print(f.groupby("label")[["n_events", "dis_ratio", "n_flips", "mean_gap", "min_gap"]].median().round(2))
f.drop(columns=["n_att", "n_dis"]).to_csv("features.csv", index=False)
print("Saved features.csv")
