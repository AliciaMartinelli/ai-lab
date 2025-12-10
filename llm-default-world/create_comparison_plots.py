"""
Create comparison plots from categorized summary CSVs.

Reads all summary CSV files from results_categorized/ and creates
side-by-side comparison plots for each attribute across all models.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

RESULTS_DIR = "results_categorized"
OUTPUT_DIR = "results_categorized"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_all_summaries():
    """Load all summary CSV files and organize by attribute."""
    summary_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith("_summary.csv")]
    
    if not summary_files:
        print(f"No summary CSV files found in {RESULTS_DIR}")
        return None
    
    all_data = defaultdict(lambda: defaultdict(dict))
    
    for summary_file in summary_files:
        csv_path = os.path.join(RESULTS_DIR, summary_file)
        try:
            df = pd.read_csv(csv_path)
            
            if "model" not in df.columns or "attribute" not in df.columns:
                print(f"SKIP {summary_file}: Missing required columns")
                continue
            
            model_name = df["model"].iloc[0]
            
            for _, row in df.iterrows():
                attr = row["attribute"]
                value = row["value"]
                percentage = row["p"] * 100
                
                if attr not in all_data:
                    all_data[attr] = {}
                
                if model_name not in all_data[attr]:
                    all_data[attr][model_name] = []
                
                all_data[attr][model_name].append({
                    "value": value,
                    "percentage": percentage,
                    "count": row["count"]
                })
        except Exception as e:
            print(f"Error reading {summary_file}: {str(e)}")
            continue
    
    return all_data


def create_comparison_plots(all_data):
    """Create side-by-side comparison plots for each attribute."""
    if not all_data:
        print("No data available for comparison plots")
        return
    
    attributes = sorted(all_data.keys())
    
    for attr in attributes:
        model_data = all_data[attr]
        
        if not model_data:
            continue
        
        models = sorted(model_data.keys())
        n_models = len(models)
        
        if n_models == 0:
            continue
        
        fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        colors = plt.cm.Set3(range(n_models))
        
        for idx, model_name in enumerate(models):
            ax = axes[idx]
            data = model_data[model_name]
            
            top_items = sorted(data, key=lambda x: x["percentage"], reverse=True)[:10]
            
            if not top_items:
                continue
            
            values = [item["value"] for item in top_items]
            percentages = [item["percentage"] for item in top_items]
            
            bars = ax.bar(values, percentages, color=colors[idx])
            ax.set_title(f"{model_name}\n{attr.replace('_', ' ').title()}", fontsize=10, fontweight='bold')
            ax.set_xlabel("Category", fontsize=9)
            ax.set_ylabel("Percentage (%)", fontsize=9)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.set_ylim(0, 105)
            
            for bar, pct in zip(bars, percentages):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{pct:.1f}%',
                       ha='center', va='bottom', fontsize=7)
        
        plt.tight_layout()
        
        comparison_path = os.path.join(OUTPUT_DIR, f"comparison_{attr}.png")
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plot saved: {comparison_path}")


if __name__ == "__main__":
    print("Loading summary CSVs...")
    all_data = load_all_summaries()
    
    if all_data:
        print(f"\nFound data for {len(all_data)} attributes")
        print("Creating comparison plots...")
        create_comparison_plots(all_data)
        print("\nDone!")
    else:
        print("No data found. Make sure summary CSV files exist in results_categorized/")

