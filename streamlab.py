# app_hierarchical_fix_mapping.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np

st.set_page_config(layout="wide", page_title="Hierarchical Learner Dashboard — Fix Mapping")
st.title("Interactive Hierarchical Learner Dashboard — Fixed Leaderboard (cummax)")

# ----------------- Config / file inputs -----------------
st.sidebar.header("Files")
mastery_upload = st.sidebar.file_uploader("Upload per_learner_mastery_long.csv (optional)", type=["csv"])
mapping_upload = st.sidebar.file_uploader("Optional: upload learner_group_map.csv", type=["csv"])

DEFAULT_MASTERY_PATHS = [
    "metrics_filtered_specific/per_learner_mastery_long.csv",
    "metrics_csvs/per_learner_mastery_long.csv",
    "per_learner_mastery_long.csv",
    "/mnt/data/per_learner_mastery_long.csv",
]
DEFAULT_MAPPING_PATHS = [
    "metrics_filtered_specific/learner_group_map_effective.csv",
    "metrics_filtered_specific/learner_group_map_clean.csv",
    "metrics_filtered_specific/learner_group_map.csv",
    "learner_group_map.csv",
    "/mnt/data/learner_group_map.csv",
]

def find_first(paths):
    for p in paths:
        if Path(p).exists():
            return p
    return None

# ----------------- Load mastery CSV -----------------
if mastery_upload is not None:
    try:
        df_master = pd.read_csv(mastery_upload, dtype=str, keep_default_na=False)
        st.sidebar.success("Loaded mastery CSV from upload")
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded mastery CSV: {e}")
        st.stop()
else:
    mpath = find_first(DEFAULT_MASTERY_PATHS)
    if not mpath:
        st.error("No mastery CSV found. Upload per_learner_mastery_long.csv or place it in expected paths.")
        st.stop()
    df_master = pd.read_csv(mpath, dtype=str, keep_default_na=False)
    st.sidebar.write(f"Loaded mastery CSV: {mpath}")

df_master.columns = [c.strip() for c in df_master.columns]

# Ensure learner_name exists
if "learner_name" not in df_master.columns:
    if "learner_idx" in df_master.columns:
        df_master["learner_name"] = df_master["learner_idx"].apply(lambda x: f"learner{int(float(x))}" if str(x).strip()!="" else "")
    elif "learner" in df_master.columns:
        df_master["learner_name"] = df_master["learner"].astype(str)
    else:
        st.error("Mastery CSV must contain 'learner_name' or 'learner_idx'.")
        st.stop()

# ----------------- Load mapping CSV (optional) -----------------
gm = None
if mapping_upload is not None:
    try:
        gm = pd.read_csv(mapping_upload, dtype=str, keep_default_na=False)
        st.sidebar.success("Loaded mapping CSV from upload")
    except Exception as e:
        st.sidebar.error(f"Failed to read mapping CSV: {e}")
        gm = None
else:
    mp = find_first(DEFAULT_MAPPING_PATHS)
    if mp:
        gm = pd.read_csv(mp, dtype=str, keep_default_na=False)
        st.sidebar.write(f"Loaded mapping CSV: {mp}")

# Simple mapping detection and clean
def norm(s): return str(s).strip().lower().replace("_","").replace(" ","")
mapping_dict = {}
if gm is not None:
    cols = [c.strip() for c in gm.columns]
    # guess columns
    learner_col = None
    group_col = None
    for c in cols:
        if "learner" in c.lower() or c.lower()=="name" or c.lower()=="id":
            learner_col = c
            break
    for c in cols:
        if "group" in c.lower() or "label" in c.lower() or "cluster" in c.lower():
            group_col = c
            break
    if learner_col is None or group_col is None:
        if len(cols) >= 2:
            learner_col, group_col = cols[0], cols[1]
    if learner_col and group_col:
        gm[learner_col] = gm[learner_col].astype(str).str.strip().str.replace('"','').str.replace("'",'')
        gm[group_col] = gm[group_col].astype(str).str.strip().str.replace('"','').str.replace("'",'')
        mapping_dict = {norm(k): v for k, v in zip(gm[learner_col], gm[group_col])}

# Detect mastery column name (mastered_count is expected)
mastered_candidates = ["mastered_count", "mastered", "value", "mastery"]
mastered_col = next((c for c in mastered_candidates if c in df_master.columns), None)
if mastered_col is None:
    st.error("Could not find mastery column. Expected one of: " + ", ".join(mastered_candidates))
    st.stop()

# Create group column on df_master (use mapping_dict)
df_master["group"] = df_master["learner_name"].apply(lambda x: mapping_dict.get(norm(x), "ungrouped"))

# Run selection
run_col = "run_id" if "run_id" in df_master.columns else "run"
run_ids = sorted(df_master[run_col].unique())
run_choice = st.sidebar.selectbox("Select run", run_ids)

# Build df_run from raw rows for that run (important)
df_run = df_master[df_master[run_col] == run_choice].copy()

# Coerce numeric types
df_run["step"] = pd.to_numeric(df_run["step"], errors="coerce")
df_run[mastered_col] = pd.to_numeric(df_run[mastered_col], errors="coerce")

# Sort in time order per learner
df_run = df_run.sort_values(["learner_name", "step"]).reset_index(drop=True)

# Deduplicate same learner+step by taking max (safe)
df_run = df_run.groupby(["learner_name", "step"], as_index=False).agg({mastered_col: "max", run_col: "first", "group": "first"})

# Now compute cummax per learner (time-ordered)
df_run = df_run.sort_values(["learner_name", "step"]).reset_index(drop=True)
df_run["mastered_cummax"] = df_run.groupby("learner_name")[mastered_col].cummax()

# --- Leaderboard computation (robust) ---
st.header("Leaderboard (latest | max | all_time_best)")

if df_run.empty:
    st.write("No data for selected run.")
else:
    # latest_logged: raw value at max step per learner
    idx_latest = df_run.groupby("learner_name")["step"].idxmax().dropna().astype(int)
    latest = df_run.loc[idx_latest, ["learner_name", "group", mastered_col, "step"]].rename(columns={mastered_col: "latest_logged", "step": "latest_step"})

    # max_logged: raw max for learner
    max_logged = df_run.groupby("learner_name", as_index=False)[mastered_col].max().rename(columns={mastered_col: "max_logged"})

    # all_time_best: max of cummax per learner
    all_best = df_run.groupby("learner_name", as_index=False)["mastered_cummax"].max().rename(columns={"mastered_cummax": "all_time_best"})

    # merge
    summary = all_best.merge(max_logged, on="learner_name", how="left").merge(latest, on="learner_name", how="left")

    # ensure group present
    if "group" not in summary.columns and "group" in df_run.columns:
        grp = df_run.groupby("learner_name", as_index=False)["group"].first()
        summary = summary.merge(grp, on="learner_name", how="left")

    # tidy ordering
    cols_want = ["learner_name", "group", "latest_logged", "latest_step", "max_logged", "all_time_best"]
    summary = summary[[c for c in cols_want if c in summary.columns]]

    # cast to int for display if numeric
    for c in ["latest_logged", "max_logged", "all_time_best"]:
        if c in summary.columns:
            summary[c] = pd.to_numeric(summary[c], errors="coerce").fillna(0).astype(int)

    summary = summary.sort_values("all_time_best", ascending=False).reset_index(drop=True)
    st.write("If `all_time_best` > `latest_logged` then the logs dropped after the best value.")
    st.dataframe(summary)


# --- Group average curves (use df_run processed) ---
st.header("Group average mastery curves")
groups = sorted(df_run["group"].unique())
fig = go.Figure()
for g in groups:
    tmp = df_run[df_run["group"] == g].copy()
    if tmp.empty:
        continue
    # If step missing, use index
    if tmp["step"].isna().all():
        tmp = tmp.reset_index(drop=True)
        tmp["step"] = tmp.index
    agg = tmp.groupby("step")[mastered_col].agg(["mean","std"]).reset_index()
    fig.add_trace(go.Scatter(x=agg["step"], y=agg["mean"], mode="lines+markers", name=f"{g}"))
    # shaded std
    fig.add_trace(go.Scatter(
        x=list(agg["step"]) + list(agg["step"][::-1]),
        y=list((agg["mean"] + agg["std"]).fillna(agg["mean"])) + list(((agg["mean"] - agg["std"]).fillna(agg["mean"]))[::-1]),
        fill="toself", opacity=0.15, showlegend=False
    ))
fig.update_layout(xaxis_title="step", yaxis_title="mastered count")
st.plotly_chart(fig, use_container_width=True)

# --- Per-learner inspect (cummax) ---
st.header("Inspect a learner (cummax)")
learner_choice = st.selectbox("Pick learner", sorted(df_run["learner_name"].unique()))
one = df_run[df_run["learner_name"] == learner_choice].sort_values("step")
one = one.reset_index(drop=True)
if one.empty:
    st.write("No rows for that learner")
else:
    one["cummax"] = one[mastered_col].cummax()
    st.write("Group:", one["group"].iloc[0])
    st.plotly_chart(px.line(one, x="step", y="cummax", markers=True, title=f"Mastery (cummax) — {learner_choice}"), use_container_width=True)
