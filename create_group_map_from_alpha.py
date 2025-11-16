# create_group_map_from_alpha.py
import pandas as pd
from sklearn.cluster import KMeans
import os

in_path = "metrics_csvs/per_learner_alpha.csv"
out_path = "metrics_csvs/learner_group_map.csv"
k = 3   # number of groups you want (choose 2..8)

df = pd.read_csv(in_path)
# compute mean alpha per learner_name across runs (if multiple runs)
mean_alpha = df.groupby("learner_name")["alpha"].mean().reset_index()
kmeans = KMeans(n_clusters=k, random_state=0).fit(mean_alpha[["alpha"]])
mean_alpha["group"] = kmeans.labels_.astype(int).astype(str)  # group ids: "0","1","2"
mean_alpha.to_csv(out_path, index=False)
print("Saved group map to", out_path)
print(mean_alpha)
