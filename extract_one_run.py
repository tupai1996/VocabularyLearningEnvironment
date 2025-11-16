# app_hierarchical_fix_mapping.py (complete, diagnostic leaderboard + cummax)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

st.set_page_config(layout="wide", page_title="Hierarchical Learner Dashboard — Fix Mapping")
st.title("Interactive Hierarchical Learner Dashboard — Fix mapping in-app")

# ---- Sidebar: file inputs ----
st.sidebar.header("Upload / select files")
mastery_upload = st.sidebar.file_uploader("Upload per_learner_mastery_long.csv (optional)", type=["csv"])
mapping_upload = st.sidebar.file_uploader("Optional: upload learner_group_map.csv", type=["csv"])

# default fallback paths the app will check
DEFAULT_MASTERY_PATHS = [
    "metrics_filtered_specific/per_learner_mastery_long.csv",
    "metrics_csvs/per_learner_mastery_long.csv",
    "per_learner_mastery_long.csv",
    "metrics_filtered_specific/per_learner_mastery_long_named.csv",
    "/mnt/data/per_learner_mastery_long.csv",
]
DEFAULT_MAPPING_PATHS = [
    "metrics_filtered_specific/learner_group_map_effective.csv",
    "metrics_filtered_specific/learner_group_map_clean.csv",
    "metrics_filtered_specific/learner_group_map.csv",
    "metrics_csvs/learner_group_map.csv",
    "learner_group_map.csv",
    "/mnt/data/learner_group_map.csv",
]

def find_first(paths):
    for p in paths:
        if Path(p).exists():
            return p
    return None

# ---- Load mastery dataframe ----
if mastery_upload is not None:
    try:
        df_master = pd.read_csv(mastery_upload, dtype=str, keep_default_na=False)
        st.sidebar.success("Loaded mastery CSV from upload")
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded mastery CSV: {e}")
        st.stop()
else:
    mpath = find_first(DEFAULT_MASTERY_PATHS)
    if mpath is None:
        st.error("No mastery CSV found. Upload per_learner_mastery_long.csv in the sidebar or place it under one of the default paths.")
        st.stop()
    df_master = pd.read_csv(mpath, dtype=str, keep_default_na=False)
    st.sidebar.write(f"Loaded mastery CSV: {mpath}")

# Normalize column names (strip)
df_master.columns = [c.strip() for c in df_master.columns]

# Ensure learner_name column exists or create from learner_idx
if "learner_name" not in df_master.columns:
    if "learner_idx" in df_master.columns:
        # handle learner_idx possibly stored as floats/strings
        df_master["learner_name"] = df_master["learner_idx"].apply(lambda x: f"learner{int(float(x))}" if str(x).strip() != "" else "")
        st.sidebar.info("Created learner_name from learner_idx")
    elif "learner" in df_master.columns:
        df_master["learner_name"] = df_master["learner"].astype(str)
    else:
        st.error("Mastery CSV must contain 'learner_name' or 'learner_idx' column. Upload a proper file.")
        st.stop()

# ---- Load mapping dataframe (if any) ----
if mapping_upload is not None:
    try:
        gm = pd.read_csv(mapping_upload, dtype=str, keep_default_na=False)
        st.sidebar.success("Loaded mapping from upload")
    except Exception as e:
        st.sidebar.error(f"Failed to load uploaded mapping CSV: {e}")
        gm = None
else:
    mapping_path = find_first(DEFAULT_MAPPING_PATHS)
    if mapping_path:
        gm = pd.read_csv(mapping_path, dtype=str, keep_default_na=False)
        st.sidebar.write(f"Loaded mapping CSV: {mapping_path}")
    else:
        gm = None
        st.sidebar.info("No mapping file found; you can create mapping here interactively.")

# ---- Helper normalize function ----
def norm(s):
    return str(s).strip().lower().replace("_","").replace(" ","")

# Attempt to detect learner and group columns in gm
def find_cols_in_mapping(gm_df):
    if gm_df is None:
        return None, None
    cols = [c.strip() for c in gm_df.columns]
    learner_candidates = ["learner_name","learner","name","id","participant","user"]
    group_candidates   = ["group","group_label","label","cluster","type","category"]
    def choose(cands):
        for cand in cands:
            for col in cols:
                if cand.lower() == col.lower():
                    return col
        # substring fallback
        for cand in cands:
            for col in cols:
                if cand.lower() in col.lower():
                    return col
        return None
    return choose(learner_candidates), choose(group_candidates)

# build initial mapping dict from gm if present
mapping_dict = {}
if gm is not None:
    learner_col, group_col = find_cols_in_mapping(gm)
    if learner_col is None or group_col is None:
        if len(gm.columns) == 2:
            learner_col, group_col = gm.columns[0], gm.columns[1]
            st.sidebar.info(f"Assuming mapping columns: {learner_col} (learner), {group_col} (group)")
        else:
            st.sidebar.warning("Could not auto-detect mapping columns. You can still assign groups below interactively.")
            learner_col, group_col = None, None

    if learner_col and group_col:
        gm[learner_col] = gm[learner_col].astype(str).str.strip().str.replace('"','').str.replace("'",'')
        gm[group_col]   = gm[group_col].astype(str).str.strip().str.replace('"','').str.replace("'",'')
        mapping_dict = {norm(k): v for k, v in zip(gm[learner_col].astype(str), gm[group_col].astype(str))}
    else:
        if len(gm.columns) >= 2:
            a, b = gm.columns[0], gm.columns[1]
            gm[a] = gm[a].astype(str).str.strip().str.replace('"','').str.replace("'",'')
            gm[b] = gm[b].astype(str).str.strip().str.replace('"','').str.replace("'",'')
            mapping_dict = {norm(k): v for k, v in zip(gm[a].astype(str), gm[b].astype(str))}
        else:
            mapping_dict = {}

# Build set of unique learners in mastery
master_names = sorted(df_master["learner_name"].astype(str).unique())
st.sidebar.write(f"Detected {len(master_names)} learners in mastery CSV")

# Apply mapping (normalized) to find matched and unmatched
matched = {}
unmatched = []
for name in master_names:
    n = norm(name)
    if n in mapping_dict:
        matched[name] = mapping_dict[n]
    else:
        unmatched.append(name)

# UI: show current mapping status
st.sidebar.subheader("Mapping status")
st.sidebar.write(f"Mapped learners: {len(matched)}")
st.sidebar.write(f"Unmapped learners: {len(unmatched)}")
if gm is not None:
    st.sidebar.write("Preview mapping (first 10):")
    try:
        st.sidebar.dataframe(gm.head(10))
    except Exception:
        st.sidebar.write("Mapping preview not available")

# ---- Interactive mapping fixer ----
st.header("Fix mapping interactively (assign groups to unmatched learners)")

with st.expander("Show unmatched learners and assign group labels", expanded=True):
    st.write("If your learners show as 'ungrouped' in the dashboard, assign groups here. Use exact group labels you want (e.g., 'fast forgetters').")
    existing_groups = sorted(set(mapping_dict.values())) if mapping_dict else []
    default_group_options = existing_groups + ["fast forgetters", "slow forgetters", "low-alpha", "mid-alpha", "high-alpha"]
    if "temp_assign" not in st.session_state:
        st.session_state.temp_assign = {}
    for name in unmatched:
        cols = st.columns([2,2,1])
        cols[0].write(f"**{name}**")
        selection = cols[1].selectbox(f"Select group for {name}", options=["<choose>"] + default_group_options, key=f"sel_{name}")
        custom = cols[2].text_input("or type custom", key=f"txt_{name}")
        final_label = None
        if selection != "<choose>":
            final_label = selection
        if custom and custom.strip() != "":
            final_label = custom.strip()
        if final_label:
            st.session_state.temp_assign[name] = final_label
        if name in st.session_state.temp_assign:
            st.write(f"Assigned: **{st.session_state.temp_assign[name]}**")

    st.markdown("---")
    if st.button("Apply mapping and save effective mapping to disk"):
        final_map = dict(mapping_dict)
        for k, v in st.session_state.temp_assign.items():
            final_map[norm(k)] = v
        effective_list = []
        for name in master_names:
            grp = final_map.get(norm(name), "ungrouped")
            effective_list.append((name, grp))
        eff_df = pd.DataFrame(effective_list, columns=["learner_name", "group"])
        save_dir = Path("metrics_filtered_specific") if Path("metrics_filtered_specific").exists() else Path(".")
        outp = save_dir / "learner_group_map_effective.csv"
        eff_df.to_csv(outp, index=False)
        st.success(f"Saved effective mapping to: {outp}")
        st.write(eff_df)
        mapping_dict = {norm(k): v for k, v in zip(eff_df["learner_name"], eff_df["group"])}
        st.session_state.temp_assign = {}

# ---- After mapping is applied or if already good, produce grouped dashboard view ----
df_master["group"] = df_master["learner_name"].apply(lambda x: mapping_dict.get(norm(x), "ungrouped"))

st.header("Dashboard — grouped view (after applying mapping)")

# choose run id if present
run_col = "run_id" if "run_id" in df_master.columns else "run"
run_ids = sorted(df_master[run_col].unique())
run_choice = st.selectbox("Select run to view", run_ids)

# select subset for the run
sub = df_master[df_master[run_col] == run_choice].copy()

# --- identify mastery column BEFORE cleaning ---
mastered_col = None
for c in ["mastered", "mastered_count", "value", "mastery"]:
    if c in sub.columns:
        mastered_col = c
        break

if mastered_col is None:
    st.error("Could not find mastery column in your mastery CSV (expected 'mastered' or 'mastered_count' or 'value').")
    st.stop()

# ---------------------------------------------------------
# CLEAN OVERLAPPING MASTERED DATA (CRITICAL FIX)
# ---------------------------------------------------------
# 1. Ensure step and mastery are numeric
sub["step"] = pd.to_numeric(sub["step"], errors="coerce")
sub[mastered_col] = pd.to_numeric(sub[mastered_col], errors="coerce")

# 2. For each learner *and* each step, keep only the max mastered value
sub_grouped_max = (
    sub.groupby(["learner_name", "step"], as_index=False)[mastered_col]
       .max()
)

# 3. Sort and merge group info back
sub_clean = sub_grouped_max.sort_values(["learner_name", "step"]).reset_index(drop=True)
sub_clean = sub_clean.merge(
    sub[["learner_name", "group"]].drop_duplicates(),
    on="learner_name",
    how="left"
)
# Ensure step numeric type for plotting
sub_clean["step"] = pd.to_numeric(sub_clean["step"], errors="coerce")
# ---------------------------------------------------------

# ---------- Leaderboard (robust, shows cummax + diagnostics) ----------
st.subheader("Leaderboard (diagnostic — latest / max / all_time_best)")

if sub_clean.empty:
    st.write("No data available after cleaning.")
else:
    # ensure numeric
    sub_clean["step"] = pd.to_numeric(sub_clean["step"], errors="coerce")
    sub_clean[mastered_col] = pd.to_numeric(sub_clean[mastered_col], errors="coerce")

    # make sure sorted
    sub_sorted = sub_clean.sort_values(["learner_name", "step"]).reset_index(drop=True)

    # compute per-learner cummax series
    sub_sorted["mastered_cummax"] = sub_sorted.groupby("learner_name")[mastered_col].cummax()

    # latest_logged: the raw mastered value at the largest step for that learner
    idx_latest = sub_sorted.groupby("learner_name")["step"].idxmax()
    idx_latest = idx_latest.dropna().astype(int)
    latest_df = sub_sorted.loc[idx_latest, ["learner_name", "group", mastered_col, "step"]].rename(
        columns={mastered_col: "latest_logged", "step": "latest_step"}
    )

    # max_logged: maximum raw mastered_count observed for that learner
    max_logged = sub_sorted.groupby("learner_name", as_index=False)[mastered_col].max().rename(
        columns={mastered_col: "max_logged"}
    )

    # all_time_best: maximum value of the cummax series (best-ever cumulative mastered)
    best = sub_sorted.groupby("learner_name", as_index=False)["mastered_cummax"].max().rename(
        columns={"mastered_cummax": "all_time_best"}
    )

    # merge into summary
    summary = best.merge(max_logged, on="learner_name", how="left").merge(latest_df, on="learner_name", how="left")

    # ensure group column present
    if "group" not in summary.columns and "group" in sub_sorted.columns:
        grp = sub_sorted.groupby("learner_name", as_index=False)["group"].first()
        summary = summary.merge(grp, on="learner_name", how="left")

    # reorder cols
    cols_order = ["learner_name", "group", "latest_logged", "latest_step", "max_logged", "all_time_best"]
    existing_cols = [c for c in cols_order if c in summary.columns]
    summary = summary[existing_cols]

    # cast to ints where appropriate for nicer display
    for c in ["latest_logged", "max_logged", "all_time_best"]:
        if c in summary.columns:
            summary[c] = pd.to_numeric(summary[c], errors="coerce").fillna(0).astype(int)

    # sort by all_time_best descending
    if "all_time_best" in summary.columns:
        summary = summary.sort_values("all_time_best", ascending=False).reset_index(drop=True)

    st.write("Summary (latest | max | all_time_best). If all_time_best > latest_logged then logs dipped after the best value.")
    st.dataframe(summary)

    # show a small table of learners that dipped (diagnostic)
    if {"latest_logged", "all_time_best"}.issubset(summary.columns):
        dips = summary[summary["latest_logged"] < summary["all_time_best"]]
        if not dips.empty:
            st.warning("These learners had their latest logged value lower than their all-time best (logging drops):")
            st.table(dips.head(20))

    # allow download of the leaderboard (all_time_best)
    csv_bytes = summary.to_csv(index=False).encode("utf-8")
    st.download_button("Download leaderboard CSV (diagnostic)", data=csv_bytes, file_name="leaderboard_diagnostic.csv", mime="text/csv")

# ---------------------------------------------------------
# Group average curves  (works with cleaned data)
# ---------------------------------------------------------
st.subheader("Group average mastery curves")

groups = sorted(sub_clean["group"].unique())
fig = go.Figure()

for g in groups:
    tmp_clean = sub_clean[sub_clean["group"] == g].copy()
    if tmp_clean.empty:
        continue

    # If step is missing, use row order as fallback
    if tmp_clean["step"].isna().all():
        tmp_clean = tmp_clean.reset_index(drop=True)
        tmp_clean["step"] = tmp_clean.index

    agg = tmp_clean.groupby("step")[mastered_col].agg(["mean","std"]).reset_index()

    # Plot mean curve
    fig.add_trace(
        go.Scatter(
            x=agg["step"],
            y=agg["mean"],
            mode="lines+markers",
            name=f"{g} (mean)"
        )
    )

    # Shaded std
    fig.add_trace(
        go.Scatter(
            x=list(agg["step"]) + list(agg["step"][::-1]),
            y=list((agg["mean"] + agg["std"]).fillna(agg["mean"])) + list(((agg["mean"] - agg["std"]).fillna(agg["mean"]))[::-1]),
            fill="toself",
            opacity=0.15,
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False
        )
    )

fig.update_layout(
    xaxis_title="step",
    yaxis_title="mastered count",
    legend_title="Group"
)

st.plotly_chart(fig, use_container_width=True)

# per-learner inspect (use sub_clean)
st.subheader("Inspect learner")
learner_choice = st.selectbox("Pick learner to inspect", sorted(sub_clean["learner_name"].unique()))
one = sub_clean[sub_clean["learner_name"] == learner_choice].sort_values("step")
st.write("Group:", one["group"].iloc[0] if not one.empty else "ungrouped")

one_plot = one.copy()
one_plot[f"{mastered_col}_cummax"] = one_plot[mastered_col].cummax()
st.plotly_chart(px.line(one_plot, x="step", y=f"{mastered_col}_cummax", markers=True, title=f"Mastery — {learner_choice} (cummax)"), use_container_width=True)
