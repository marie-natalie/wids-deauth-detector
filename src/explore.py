import sys, pandas as pd
df = pd.read_csv(sys.argv[1], low_memory=False)
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nDtypes:\n", df.dtypes.value_counts())
print("\nFirst 3 rows:\n", df.head(3).T)
print("\nMissing values (top 10):\n", df.isna().sum().sort_values(ascending=False).head(10))
for c in df.columns:
    if df[c].nunique() < 15:
        print(f"\n{c}:\n{df[c].value_counts()}")
