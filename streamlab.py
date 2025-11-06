import streamlit as st
import pandas as pd
import plotly.express as px

# Load your prepared CSV (long format)
df = pd.read_csv("per_learner_mastery_long.csv")

st.title("📊 Learner Mastery Progression")

# Pick a run
run_id = st.selectbox("Select Run", sorted(df["run_id"].unique()))
sub = df[df["run_id"] == run_id]

# Leaderboard
last = sub.loc[sub.groupby("learner_name")["step"].idxmax()]
st.subheader("🏆 Leaderboard")
st.dataframe(last[["learner_name", "mastered", "step"]].sort_values("mastered", ascending=False))

# Per-learner curve
choice = st.sidebar.selectbox("Select learner", sorted(sub["learner_name"].unique()))
one = sub[sub["learner_name"] == choice]
st.plotly_chart(px.line(one, x="step", y="mastered", markers=True, title=f"Mastery: {choice}"), use_container_width=True)

# All learners overlay
st.subheader("🧩 All learners")
st.plotly_chart(px.line(sub, x="step", y="mastered", color="learner_name"), use_container_width=True)
