"""Generate and save the final cross-dataset results table."""
import pandas as pd
from pathlib import Path

CSV_PATH = Path("results/cross_dataset_results.csv")
df = pd.read_csv(CSV_PATH)

metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]

lines = []
lines.append("=" * 100)
lines.append("CROSS-DATASET GENERALIZATION RESULTS")
lines.append("=" * 100)

for scenario, train_ds in [("Scenario A: Trained on CelebDF", "CelebDF"),
                            ("Scenario B: Trained on UADFV",   "UADFV")]:
    lines.append(f"\n{scenario}")
    lines.append("-" * 100)
    header = f"{'Test Dataset':<12} {'Backbone':<10}" + "".join(f"  {m:<11}" for m in metrics)
    lines.append(header)
    lines.append("-" * 100)
    sub = df[df["Train Dataset"] == train_ds].sort_values(["Test Dataset", "Backbone"])
    prev_test = None
    for _, row in sub.iterrows():
        if row["Test Dataset"] != prev_test and prev_test is not None:
            lines.append("")
        prev_test = row["Test Dataset"]
        vals = "".join(f"  {row[m]:<11.4f}" for m in metrics)
        lines.append(f"{row['Test Dataset']:<12} {row['Backbone']:<10}{vals}")

lines.append("")
lines.append("=" * 100)

table_text = "\n".join(lines)
print(table_text)

out = Path("results/cross_dataset_table.txt")
out.write_text(table_text)
print(f"\nSaved to {out}")
