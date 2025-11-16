# clean_group_map.py
import pandas as pd
from pathlib import Path

in_path = "metrics_filtered_specific/learner_group_map.csv"   # change if needed
out_path = "metrics_filtered_specific/learner_group_map_clean.csv"

p = Path(in_path)
if not p.exists():
    raise SystemExit(f"Input file not found: {in_path}")

# Read robustly
df = pd.read_csv(p, header=0, engine='python', dtype=str, keep_default_na=False)

# Clean column names (strip quotes/whitespace)
df.columns = [c.strip().replace('"', '').replace("'", "").strip() for c in df.columns]

# If the first column currently contains numeric indices (0..7) and there's no 'learner_name' column,
# convert indices to learner names 'learner0'..'learner7'
first_col = df.columns[0]
if 'learner_name' not in df.columns:
    # check whether the first column looks like indices
    col_vals = df[first_col].tolist()
    is_index_like = all(v.isdigit() or v=='' for v in col_vals)
    if is_index_like:
        df['learner_name'] = [f"learner{int(v)}" if v!='' else '' for v in col_vals]
    else:
        # otherwise assume first column is learner identifier but may need stripping
        df['learner_name'] = df[first_col].astype(str).str.strip().str.replace('"','').str.replace("'",'')

# Find the group column among common candidates
candidates = ['group','group_label','label','cluster','type']
group_col = None
for c in candidates:
    if c in df.columns:
        group_col = c
        break
# If not found, try any remaining column that is not learner_name
if group_col is None:
    alt = [c for c in df.columns if c != 'learner_name']
    if len(alt) == 1:
        group_col = alt[0]
    else:
        raise SystemExit(f"Could not auto-detect group column. Found columns: {list(df.columns)}. Please ensure there is a group column (e.g. 'group' or 'group_label').")

# Build cleaned mapping DataFrame and save
clean = df[['learner_name', group_col]].rename(columns={group_col: 'group'})
clean['learner_name'] = clean['learner_name'].astype(str).str.strip()
clean['group'] = clean['group'].astype(str).str.strip()

clean.to_csv(out_path, index=False)
print("Saved cleaned mapping to:", out_path)
print(clean)
