"""
Analysis script for manually categorized responses.

Expects CSV files in the 'outputs' directory with a filled 'Answer_Cat' column.
Generates summaries, plots, and a final overview per model.
"""

import os
import math
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import norm

OUTPUTS_DIR = "outputs"
RESULTS_DIR = "results_categorized"
os.makedirs(RESULTS_DIR, exist_ok=True)


def proportion_ci(k, n, alpha=0.05):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    se = math.sqrt(p * (1 - p) / n)
    z = norm.ppf(1 - alpha / 2)
    return p, p - z * se, p + z * se


def analyze_categorized_csv(csv_path):
    """Analyze a categorized CSV file (with Answer_Cat column)."""
    df = pd.read_csv(csv_path)
    if "Answer_Cat" not in df.columns:
        print(f"SKIP {csv_path}: Column 'Answer_Cat' missing.")
        return None

    model_name = df["Model"].iloc[0] if "Model" in df.columns else os.path.basename(csv_path).replace(".csv", "")

    if df["Answer_Cat"].isna().all() or (df["Answer_Cat"] == "").all():
        print(f"SKIP {csv_path}: 'Answer_Cat' is empty. Please fill it first.")
        return None

    summary_rows = []
    attributes = df["Attribute"].unique()

    for attr in attributes:
        attr_df = df[df["Attribute"] == attr]
        cats = attr_df["Answer_Cat"].fillna("").astype(str)
        cats = cats[cats != ""]
        if len(cats) == 0:
            continue

        total = len(cats)
        counter = Counter(cats)
        top_items = counter.most_common(10)

        print(f"\n{model_name} → {attr}")
        for value, count in top_items:
            p, low, high = proportion_ci(count, total)
            print(f"  {value:20s}: {count:3d} ({p*100:5.1f}%  CI [{low*100:5.1f}%, {high*100:5.1f}%])")
            summary_rows.append(
                {
                    "model": model_name,
                    "attribute": attr,
                    "value": value,
                    "count": count,
                    "p": p,
                    "ci_low": low,
                    "ci_high": high,
                }
            )

        values = [v for v, _ in top_items]
        counts = [c for _, c in top_items]
        percentages = [(c / total) * 100 for c in counts]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.bar(values, counts, color="steelblue")
        ax1.set_title(f"{model_name} – {attr} (Counts)")
        ax1.set_xlabel("Category")
        ax1.set_ylabel("Count")
        ax1.tick_params(axis="x", rotation=45)

        ax2.bar(values, percentages, color="coral")
        ax2.set_title(f"{model_name} – {attr} (Percentages)")
        ax2.set_xlabel("Category")
        ax2.set_ylabel("Percentage (%)")
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, f"{model_name.replace('/', '_').replace(':', '_')}_{attr}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(RESULTS_DIR, f"{model_name.replace('/', '_').replace(':', '_')}_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved: {summary_path}")

    return summary_rows


def create_final_summary(all_summaries):
    """Create a final overview (CSV + plot) across all models/attributes."""
    if not all_summaries:
        print("No data available for final summary.")
        return

    summary_df = pd.DataFrame(all_summaries)
    final_rows = []

    for attr in summary_df["attribute"].unique():
        attr_df = summary_df[summary_df["attribute"] == attr]
        for model in attr_df["model"].unique():
            m_df = attr_df[attr_df["model"] == model]
            if m_df.empty:
                continue
            top_row = m_df.sort_values("p", ascending=False).iloc[0]
            final_rows.append(
                {
                    "attribute": attr,
                    "model": model,
                    "top_value": top_row["value"],
                    "percentage": top_row["p"] * 100,
                }
            )

    if not final_rows:
        print("No final rows generated.")
        return

    final_df = pd.DataFrame(final_rows)
    final_csv = os.path.join(RESULTS_DIR, "final_summary.csv")
    final_df.to_csv(final_csv, index=False)
    print(f"Final summary CSV saved: {final_csv}")

    attributes = list(final_df["attribute"].unique())
    models = list(final_df["model"].unique())
    n_attr = len(attributes)

    fig, axes = plt.subplots(n_attr, 1, figsize=(14, 3 * n_attr))
    if n_attr == 1:
        axes = [axes]

    colors = plt.cm.Set3(range(len(models)))

    for idx, attr in enumerate(attributes):
        ax = axes[idx]
        a_df = final_df[final_df["attribute"] == attr].sort_values("percentage", ascending=True)
        y_pos = range(len(a_df))
        bars = ax.barh(y_pos, a_df["percentage"], color=colors[: len(a_df)])
        labels = [f"{row['model']}: {row['top_value']}" for _, row in a_df.iterrows()]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Percentage (%)", fontsize=10)
        ax.set_title(attr.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_xlim(0, 105)
        for bar, pct in zip(bars, a_df["percentage"]):
            ax.text(pct + 1, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%", va="center", fontsize=8)
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    final_png = os.path.join(RESULTS_DIR, "final_summary.png")
    plt.savefig(final_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Final summary plot saved: {final_png}")


if __name__ == "__main__":
    csv_files = [f for f in os.listdir(OUTPUTS_DIR) if f.endswith(".csv")]
    if not csv_files:
        print(f"No CSV files found in {OUTPUTS_DIR}.")
        exit(1)

    all_summaries = []
    for csv_file in csv_files:
        csv_path = os.path.join(OUTPUTS_DIR, csv_file)
        summary = analyze_categorized_csv(csv_path)
        if summary:
            all_summaries.extend(summary)

    create_final_summary(all_summaries)

