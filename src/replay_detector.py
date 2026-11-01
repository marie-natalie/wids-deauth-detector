import argparse, json, time, collections
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

FEATURES = ["n_events", "dis_ratio", "n_flips", "mean_gap", "min_gap", "max_gap",
            "hist_60s", "hist_300s", "hist_flips_300s"]
CSV = "Wi-Fi-Association_Disassociation-Dataset/Aassociation-Disassociation_Dataset.csv"
S = pd.Timedelta

ap = argparse.ArgumentParser()
ap.add_argument("--day", help="day to replay, YYYY-MM-DD")
ap.add_argument("--speed", type=float, default=0, help="0 = instant; 600 = 600 dataset-seconds per real second")
ap.add_argument("--max-sleep", type=float, default=0.3, help="longest pause between windows (s)")
args = ap.parse_args()

feat = pd.read_csv("features2.csv")
if not args.day:
    t = feat.groupby("day")["label"].agg(attack_windows="sum", total_windows="size")
    print(t[t.attack_windows > 0].to_string())
    raise SystemExit("\nPick one day (use one from 2022-10-11 onward) with --day YYYY-MM-DD")

# 1. Train on every OTHER day, so the replayed day is unseen
train = feat[feat["day"] != args.day]
model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model.fit(train[FEATURES], train["label"])
print(f"Model trained on {len(train)} windows from other days; it has never seen {args.day}.\n")

# 2. Load the events for that day (+10 min warm-up so the history features are filled)
df = pd.read_csv(CSV)
df["ts"] = pd.to_datetime(df["@timestamp"], format="%b %d, %Y @ %H:%M:%S.%f")
df = df.sort_values("ts", kind="stable").reset_index(drop=True)
d0 = pd.Timestamp(args.day)
ev = df[(df.ts >= d0 - S(minutes=10)) & (df.ts < d0 + S(days=1))]

last_ts, last_dis, cur = {}, {}, {}
hist = collections.defaultdict(collections.deque)    # per device: (window, events, flips)
flags = collections.defaultdict(collections.deque)   # per device: recent flagged windows
blocklist, dev_att = {}, collections.Counter()
streamed, y_true, y_pred = [], [], []
n_alerts = 0

def flush(win):
    """A 10-second window just closed: build features per device, classify, act."""
    global n_alerts
    devs, rows = list(cur), []
    for dev in devs:
        c, h = cur[dev], hist[dev]
        while h and h[0][0] < win - S(seconds=300):
            h.popleft()
        h60 = sum(n for w, n, f in h if w >= win - S(seconds=60))
        h300 = sum(n for w, n, f in h)
        hf = sum(f for w, n, f in h)
        g = c["gaps"]
        rows.append([c["n"], c["dis"] / c["n"], c["flips"], sum(g) / len(g), min(g), max(g), h60, h300, hf])
        h.append((win, c["n"], c["flips"]))
    if win >= d0 and rows:
        preds = model.predict(pd.DataFrame(rows, columns=FEATURES))
        for dev, r, p in zip(devs, rows, preds):
            truth = int(cur[dev]["att"] > 0)
            dev_att[dev] += truth
            streamed.append(dict(mac_address=dev, win=win, **dict(zip(FEATURES, r))))
            y_true.append(truth); y_pred.append(int(p))
            if p == 1:
                n_alerts += 1
                q = flags[dev]
                while q and q[0] < win - S(seconds=60):
                    q.popleft()
                if not q:
                    print(f"[ALERT] {win}  {dev}: {int(r[0])} events in 10 s, {int(r[7])} in the last 5 min")
                q.append(win)
                if len(q) >= 2 and dev not in blocklist:          # WIPS action
                    blocklist[dev] = {"blocked_at": str(win), "reason": "repeated disconnect flood"}
                    json.dump(blocklist, open("blocklist.json", "w"), indent=2)
                    print(f"[WIPS]  {win}  BLOCKED {dev}  -> blocklist.json")
    cur.clear()

# 3. Replay the events in time order
cur_win = None
for r in ev.itertuples(index=False):
    dev, ts = r.mac_address, r.ts
    dis, att = int(r.association == "disassociated"), int(r.dataType == "attack")
    w = ts.floor("10s")
    if cur_win is None:
        cur_win = w
    if w > cur_win:
        flush(cur_win)
        if args.speed > 0 and w >= d0:
            time.sleep(min((w - cur_win).total_seconds() / args.speed, args.max_sleep))
        cur_win = w
    gap = 3600.0 if dev not in last_ts else min((ts - last_ts[dev]).total_seconds(), 3600.0)
    flip = int(dev in last_dis and last_dis[dev] != dis)
    c = cur.setdefault(dev, dict(n=0, dis=0, flips=0, gaps=[], att=0))
    c["n"] += 1; c["dis"] += dis; c["flips"] += flip; c["gaps"].append(gap); c["att"] += att
    last_ts[dev], last_dis[dev] = ts, dis
flush(cur_win)

# 4. Summary
print(f"\n=== Replay summary for {args.day} ===")
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
print("Confusion matrix [[TN FP] [FN TP]]:\n", cm)
print(f"Windows: {cm.sum()} | accuracy: {(tn + tp) / cm.sum():.4f} | flagged windows: {n_alerts}")
print("Blocked devices:", {d: f"{dev_att[d]} true attack windows that day" for d in blocklist} or "none")

s = pd.DataFrame(streamed)
off = feat[feat.day == args.day].copy()
off["win"] = pd.to_datetime(off["win"])
m = off.merge(s, on=["mac_address", "win"], suffixes=("_off", "_str"))
diff = max(float((m[f + "_off"] - m[f + "_str"]).abs().max()) for f in FEATURES)
print(f"\nFEATURE CHECK: {len(m)}/{len(off)} offline windows reproduced by the stream, max difference {diff:.2e}")
for f in FEATURES:
    d = (m[f+"_off"] - m[f+"_str"]).abs()
    print(f"  {f:16s} max diff {d.max():.2e}  windows differing: {(d > 1e-9).sum()}")
