"""Check NST data quality in features."""

import pandas as pd

df = pd.read_csv('data/game_features.csv')

nst_cols = [c for c in df.columns if any(x in c.lower() for x in ['xgf', 'cf', 'hdcf', 'scf', 'pdo'])]

print("=" * 60)
print("NST Feature Data Quality Check")
print("=" * 60)
print(f"\nTotal games: {len(df)}")
print(f"NST feature columns: {len(nst_cols)}")

print("\nNaN counts for NST features:")
for col in nst_cols:
    nan_count = df[col].isna().sum()
    pct = nan_count / len(df) * 100
    print(f"  {col}: {nan_count}/{len(df)} ({pct:.1f}% missing)")

# Check how many games have ALL NST features vs NONE
games_with_all_nst = df[nst_cols].notna().all(axis=1).sum()
games_with_no_nst = df[nst_cols].isna().all(axis=1).sum()
games_with_some_nst = len(df) - games_with_all_nst - games_with_no_nst

print(f"\nGames with ALL NST features: {games_with_all_nst} ({games_with_all_nst/len(df)*100:.1f}%)")
print(f"Games with SOME NST features: {games_with_some_nst} ({games_with_some_nst/len(df)*100:.1f}%)")
print(f"Games with NO NST features: {games_with_no_nst} ({games_with_no_nst/len(df)*100:.1f}%)")

# Check date range where NST data is available
if games_with_all_nst > 0:
    games_with_nst = df[df[nst_cols].notna().all(axis=1)]
    print(f"\nDate range with NST data:")
    print(f"  From: {games_with_nst['date'].min()}")
    print(f"  To: {games_with_nst['date'].max()}")

