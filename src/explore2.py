import pandas as pd
F = "Wi-Fi-Association_Disassociation-Dataset/Aassociation-Disassociation_Dataset.csv"
df = pd.read_csv(F)
df["ts"] = pd.to_datetime(df["@timestamp"], format="%b %d, %Y @ %H:%M:%S.%f")
df = df.sort_values("ts").reset_index(drop=True)

print("=== Date range per label ===")
print(df.groupby("dataType")["ts"].agg(["min", "max", "count"]))

print("\n=== Rows per device per label ===")
print(pd.crosstab(df["mac_address"], df["dataType"]))

print("\n=== Association state per label ===")
print(pd.crosstab(df["dataType"], df["association"], normalize="index").round(3))

df["gap"] = df.groupby("mac_address")["ts"].diff().dt.total_seconds()
print("\n=== Seconds between events (same device), by label ===")
print(df.groupby("dataType")["gap"].describe(percentiles=[.1, .5, .9]).round(2))

df["win"] = df["ts"].dt.floor("10s")
w = df.groupby(["mac_address", "win", "dataType"]).size().reset_index(name="events")
print("\n=== Events per device per 10-second window, by label ===")
print(w.groupby("dataType")["events"].describe(percentiles=[.1, .5, .9]).round(2))
