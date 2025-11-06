from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import os

logdir = "runs"
out_csv = "all_metrics.csv"

all_dfs = []

for root, dirs, files in os.walk(logdir):
    for f in files:
        if f.startswith("events.out"):
            subset = df[df["run"].str.contains("ab_balanced")]
            ea = event_accumulator.EventAccumulator(os.path.join(root, f))
            ea.Reload()
            tags = ea.Tags()["scalars"]
            for tag in tags:
                events = ea.Scalars(tag)
                df = pd.DataFrame([(e.step, e.value, tag, root) for e in events],
                                  columns=["step", "value", "metric", "run"])
                all_dfs.append(df)

final = pd.concat(all_dfs)
final.to_csv(out_csv, index=False)
print(f"✅ Saved all metrics to {out_csv}")
