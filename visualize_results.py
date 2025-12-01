import pandas as pd
import matplotlib.pyplot as plt
import os

ALL_RUNS_CSV = "all_runs.csv"
SUMMARY_CSV = "summary_best.csv"

all_runs = pd.read_csv(ALL_RUNS_CSV)
summary = pd.read_csv(SUMMARY_CSV)

out_dir = "plots"
os.makedirs(out_dir, exist_ok=True)

plt.figure(figsize=(8, 5))
models = all_runs["model"].unique()
data = [all_runs.loc[all_runs["model"] == m, "f1"].dropna().values for m in models]
plt.boxplot(data, labels=models, patch_artist=False)
plt.title("Phân phối F1-score theo từng mô hình")
plt.ylabel("F1-score")
plt.xlabel("Model")
plt.grid(axis="y", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "boxplot_f1_per_model.png"))
plt.show()


plt.figure(figsize=(10, 6))
for m in models:
    dfm = all_runs[all_runs["model"] == m].sort_values("run")
    plt.plot(dfm["run"], dfm["accuracy"], marker="o", label=m)
plt.title("Độ chính xác cho mỗi lần chạy của từng mô hình")
plt.xlabel("Run")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "accuracy_per_run.png"))
plt.show()

plt.figure(figsize=(10, 6))
metrics = ["accuracy", "precision", "recall", "f1"]
x = range(len(metrics))
width = 0.15

for idx, m in enumerate(models):
    row = summary[summary["model"] == m]
    if row.empty:
        continue

    means = [
        row["avg_accuracy"].values[0] if "avg_accuracy" in row else 0.0,
        row["avg_precision"].values[0] if "avg_precision" in row else 0.0,
        row["avg_recall"].values[0] if "avg_recall" in row else 0.0,
        row["avg_f1"].values[0] if "avg_f1" in row else 0.0,
    ]
    stds = [
        row["std_accuracy"].values[0] if "std_accuracy" in row else 0.0,
        row["std_precision"].values[0] if "std_precision" in row else 0.0,
        row["std_recall"].values[0] if "std_recall" in row else 0.0,
        row["std_f1"].values[0] if "std_f1" in row else 0.0,
    ]

    xpos = [p + idx * width for p in x]
    plt.bar(xpos, means, width=width, yerr=stds, capsize=4, label=m)


plt.xticks([p + width * (len(models) - 1) / 2 for p in x], metrics)
plt.ylabel("Score")
plt.title("Số liệu trung bình ± độ lệch chuẩn trên mỗi mô hình")
plt.legend()
plt.grid(axis="y", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "avg_metrics_per_model.png"))
plt.show()

plt.figure(figsize=(8, 6))
for m in models:
    dfm = all_runs[all_runs["model"] == m]
    plt.scatter(dfm["accuracy"], dfm["f1"], s=40, label=m)
plt.xlabel("Accuracy")
plt.ylabel("F1-score")
plt.title("F1 vs Accuracy (mỗi điểm tương ứng với một lần chạy)")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "f1_vs_accuracy_scatter.png"))
plt.show()

print("Saved plots into:", out_dir)
