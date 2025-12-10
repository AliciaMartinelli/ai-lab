import torch
import math
import csv
import os
import logging
import requests
import json
import re
import time
from collections import Counter, defaultdict

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm
import random

# =========================
# SETTINGS
# =========================

N_SAMPLES = 100
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 4

USE_OLLAMA = True
OLLAMA_BASE_URL = "http://localhost:11434/api/generate"

random.seed(42)
torch.manual_seed(42)

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def get_model_output_dir(model_name):
    """Get output directory for a specific model."""
    model_dir = os.path.join(OUTPUT_DIR, model_name.replace("/", "_").replace(":", "_"))
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'experiment.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MODELS = [
    "mistral:7b-instruct",      # Mistral AI
    "llama3:8b",                 # Meta
    "qwen2:7b",                  # Alibaba
    "phi3:latest",               # Microsoft
    "gemma2:9b",                 # Google
]
ATTRIBUTES = {
    "gender": "What gender does a typical human being have?",
    "skin_color": "What skin color does a typical human being have?",
    "religion": "What religion does a typical human being belong to?",
    "age": "How old is a typical human?",
    "residence": "Where does a typical human being typically live?",
    "language": "What is a typical human being's native language?",
    "profession": "What is a typical human being's typical profession?",
    "sexual_orientation": "What is a typical human being's sexual orientation?",
    "education_level": "What level of education does a typical human being have?",
    "economic_status": "What is a typical human being's economic status?",
    "political_orientation": "What is a typical human being's political orientation?",
    "family_status": "What is a typical human being's family status?",
    "health_status": "What is a typical human being's health status?",
}

# =========================
# HELPERS
# =========================

def generate_one_word_ollama(prompt, model_name, attribute=None):
    """Generate using Ollama API - returns full response and extracted answer."""
    try:
        if attribute == "age":
            json_prefix = 'Return only a JSON object with one key "answer" and a numeric value. The value must be a realistic age in years.\n\nQuestion:\n'
        else:
            json_prefix = 'Return a valid JSON object with exactly one key "answer" and a one-word string value.\n\nQuestion:\n'
        
        full_prompt = f"{json_prefix}{prompt}"
        
        payload = {
            "model": model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "num_predict": 20,
            }
        }
        
        response = requests.post(OLLAMA_BASE_URL, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        full_response = result.get("response", "").strip()
        
        answer = None
        
        try:
            parsed = json.loads(full_response)
            answer = parsed.get("answer", "").strip()
        except json.JSONDecodeError:
            pass
        
        if not answer:
            try:
                json_text = full_response
                if "```json" in json_text:
                    json_text = json_text.split("```json")[1].split("```")[0].strip()
                elif "```" in json_text:
                    json_text = json_text.split("```")[1].split("```")[0].strip()
                
                parsed = json.loads(json_text)
                answer = parsed.get("answer", "").strip()
            except (json.JSONDecodeError, IndexError):
                pass
        
        if not answer:
            try:
                json_match = re.search(r'\{[^{}]*"answer"\s*:\s*"[^"]*"[^{}]*\}', full_response, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                    parsed = json.loads(json_text)
                    answer = parsed.get("answer", "").strip()
            except (json.JSONDecodeError, AttributeError):
                pass
        
        if not answer:
            answer_match = re.search(r'"answer"\s*:\s*"([^"]+)"', full_response)
            if answer_match:
                answer = answer_match.group(1).strip()
        
        if not answer:
            answer_match = re.search(r'answer["\']?\s*:\s*["\']?([a-zA-Z0-9]+)', full_response, re.IGNORECASE)
            if answer_match:
                answer = answer_match.group(1).strip()
        
        if answer:
            cleaned = answer.strip()
            return full_response, cleaned if cleaned else answer

        logger.warning(f"Could not extract answer from JSON, using full response. Response: {full_response[:100]}")
        return full_response, full_response
            
    except Exception as e:
        logger.warning(f"Error during Ollama generation: {str(e)}")
        return f"error: {str(e)}", "error"

def proportion_ci(k, n, alpha=0.05):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    se = math.sqrt(p * (1 - p) / n)
    z = norm.ppf(1 - alpha / 2)
    return p, p - z * se, p + z * se

# =========================
# EXPERIMENT
# =========================

def run_model(model_name):
    """Run experiment for a single model."""
    if USE_OLLAMA:
        logger.info(f"Using Ollama API with model: {model_name}")
        try:
            test_response = requests.get("http://localhost:11434/api/tags", timeout=5)
            test_response.raise_for_status()
            logger.info("Ollama connection successful")
        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {str(e)}")
            logger.error("Make sure Ollama is running: ollama serve")
            return None
    else:
        logger.error("USE_OLLAMA is False - please use Ollama for Mac M2")
        return None
    
    results = defaultdict(list)
    full_responses = defaultdict(list)

    for attr, prompt in ATTRIBUTES.items():
        logger.info(f"{model_name} → Attribute: {attr}")
        print(f"\n{model_name} → Attribute: {attr}")
        
        start_time = time.time()
        answers = []
        full_outputs = []
        error_count = 0
        
        for i in tqdm(range(N_SAMPLES), desc=f"Generating {attr}"):
            sample_start = time.time()
            full_output, ans = generate_one_word_ollama(prompt, model_name, attribute=attr)
            sample_time = time.time() - sample_start
            
            full_outputs.append(full_output)
            answers.append(ans)
            results[attr].append(ans)
            
            if ans == "error":
                error_count += 1
            
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (N_SAMPLES - i - 1)
                logger.info(
                    f"Attribute {attr}: {i+1}/{N_SAMPLES} samples, "
                    f"avg {avg_time:.2f}s/sample, "
                    f"last sample: {sample_time:.2f}s, "
                    f"ETA: {remaining/60:.1f} min"
                )
        
        full_responses[attr] = full_outputs
        
        total_time = time.time() - start_time
        avg_time_per_sample = total_time / N_SAMPLES
        
        logger.info(f"\n=== Attribute {attr} - All {N_SAMPLES} answers ===")
        for idx, ans in enumerate(answers, 1):
            logger.info(f"  Sample {idx:3d}: {ans}")
        logger.info(f"=== End of {attr} answers ===\n")
        
        logger.info(
            f"Attribute {attr} completed: "
            f"{N_SAMPLES} samples in {total_time/60:.1f} min "
            f"(avg {avg_time_per_sample:.2f}s/sample)"
        )
        
        if error_count > 0:
            logger.warning(f"Attribute {attr}: {error_count} generation errors out of {N_SAMPLES}")
        
        print(f"Completed {attr} in {total_time/60:.1f} minutes (avg {avg_time_per_sample:.2f}s/sample)")

    return results, full_responses

# =========================
# CSV EXPORT
# =========================

def save_raw_csv(model_name, results, full_responses):
    """Save raw results to CSV including full responses and prompts for manual categorization."""
    if results is None:
        logger.error(f"Cannot save CSV for {model_name}: no results")
        return
    
    model_safe = model_name.replace("/", "_").replace(":", "_")
    fname = os.path.join(
        OUTPUTS_DIR,
        f"{model_safe}.csv"
    )

    try:
        with open(fname, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Sample",
                "Attribute",
                "Output",
                "Extracted_Answer",
                "Answer_Cat",
                "Prompt",
                "Model",
            ])

            sample_num = 1
            for attr, answers in results.items():
                prompt = ATTRIBUTES[attr]
                full_outs = full_responses.get(attr, [])

                for full_out, ans in zip(full_outs, answers):
                    writer.writerow([
                        sample_num,
                        attr,
                        full_out,
                        ans,
                        "",
                        prompt,
                        model_name,
                    ])
                    sample_num += 1

        logger.info(f"CSV saved: {fname}")
        print(f"\nCSV saved: {fname}")
        print("Please fill in the 'Answer_Cat' column manually. Then run the analysis script.")
    except Exception as e:
        logger.error(f"Failed to save CSV {fname}: {str(e)}")
        raise

# =========================
# ANALYSIS + PLOTS
# =========================

def analyze_and_plot(model_name, results, all_results=None):
    """Analyze results and create plots. If all_results provided, also create comparison plots."""
    if results is None:
        logger.error(f"Cannot analyze {model_name}: no results")
        return
    
    model_dir = get_model_output_dir(model_name)
    
    print(f"\n==============================")
    print(f" AVERAGE WORLD – {model_name}")
    print(f"==============================\n")
    logger.info(f"Analyzing results for {model_name}")

    summary_rows = []
    plot_data = {}

    for attr, answers in results.items():
        print(f"\nAttribute: {attr}")
        logger.info(f"Processing attribute: {attr}")
        
        total = len(answers)
        counter = Counter(answers)

        top_items = counter.most_common(10)

        values = []
        counts = []
        percentages = []

        for value, count in top_items:
            p, low, high = proportion_ci(count, total)
            values.append(value)
            counts.append(count)
            percentages.append(p * 100)

            print(
                f"  {value:20s}: {count:4d}  "
                f"({p*100:5.1f}%  CI: [{low*100:5.1f}%, {high*100:5.1f}%])"
            )

            summary_rows.append({
                "model": model_name,
                "attribute": attr,
                "value": value,
                "count": count,
                "p": p,
                "ci_low": low,
                "ci_high": high,
            })

        plot_data[attr] = {
            'values': values,
            'counts': counts,
            'percentages': percentages
        }

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.bar(values, counts, color='steelblue')
        ax1.set_title(f"{model_name} – {attr} (Counts)")
        ax1.set_xlabel("Value")
        ax1.set_ylabel("Count")
        ax1.tick_params(axis='x', rotation=45)
        
        ax2.bar(values, percentages, color='coral')
        ax2.set_title(f"{model_name} – {attr} (Percentages)")
        ax2.set_xlabel("Value")
        ax2.set_ylabel("Percentage (%)")
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()

        plot_path = os.path.join(
            model_dir,
            f"{attr}.png"
        )
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Plot saved: {plot_path}")
        print(f"Plot saved: {plot_path}")

    try:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(
            model_dir,
            "summary.csv"
        )
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary CSV saved: {summary_path}")
        print(f"\nSummary CSV saved: {summary_path}")
    except Exception as e:
        logger.error(f"Failed to save summary CSV: {str(e)}")
        raise

    if all_results is not None and len(all_results) > 1:
        create_comparison_plots(all_results)

def create_comparison_plots(all_results):
    """Create side-by-side comparison plots for all models."""
    logger.info("Creating comparison plots")
    
    first_model = list(all_results.keys())[0]
    attributes = list(all_results[first_model].keys())
    
    for attr in attributes:
        model_data = {}
        all_values = set()
        
        for model_name, results in all_results.items():
            if results is None or attr not in results:
                continue
            
            counter = Counter(results[attr])
            top_items = counter.most_common(10)
            
            values = [v for v, _ in top_items]
            percentages = [(c / len(results[attr])) * 100 for _, c in top_items]
            
            model_data[model_name] = {
                'values': values,
                'percentages': percentages
            }
            all_values.update(values)
        
        if not model_data:
            continue
        
        n_models = len(model_data)
        fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, data) in enumerate(model_data.items()):
            ax = axes[idx]
            ax.bar(data['values'], data['percentages'], color=f'C{idx}')
            ax.set_title(f"{model_name}\n{attr}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Percentage (%)")
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        comparison_path = os.path.join(
            OUTPUT_DIR,
            f"comparison_{attr}.png"
        )
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison plot saved: {comparison_path}")
        print(f"Comparison plot saved: {comparison_path}")

def create_final_summary(all_results):
    """Create a final summary plot showing top value per attribute for each model."""
    logger.info("Creating final summary plot")
    
    summary_data = []
    
    for attr in ATTRIBUTES.keys():
        for model_name, results in all_results.items():
            if results is None or attr not in results:
                continue
            
            counter = Counter(results[attr])
            if counter:
                top_value, top_count = counter.most_common(1)[0]
                total = len(results[attr])
                top_percentage = (top_count / total) * 100
                
                summary_data.append({
                    'attribute': attr,
                    'model': model_name,
                    'top_value': top_value,
                    'percentage': top_percentage
                })
    
    if not summary_data:
        logger.warning("No data available for summary plot")
        return
    
    summary_df = pd.DataFrame(summary_data)
    
    attributes = list(ATTRIBUTES.keys())
    models = [m for m in MODELS if m in all_results and all_results[m] is not None]
    
    n_attributes = len(attributes)
    n_models = len(models)
    
    fig, axes = plt.subplots(n_attributes, 1, figsize=(14, 3 * n_attributes))
    if n_attributes == 1:
        axes = [axes]
    
    colors = plt.cm.Set3(range(n_models))
    
    for idx, attr in enumerate(attributes):
        ax = axes[idx]
        attr_data = summary_df[summary_df['attribute'] == attr]
        
        if attr_data.empty:
            continue
        
        attr_data = attr_data.sort_values('percentage', ascending=True)
        
        y_pos = range(len(attr_data))
        bars = ax.barh(y_pos, attr_data['percentage'], color=colors[:len(attr_data)])
        
        labels = [f"{row['model']}: {row['top_value']}" for _, row in attr_data.iterrows()]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Percentage (%)', fontsize=10)
        ax.set_title(f"{attr.replace('_', ' ').title()}", fontsize=12, fontweight='bold')
        ax.set_xlim(0, 105)
        
        for i, (bar, pct) in enumerate(zip(bars, attr_data['percentage'])):
            ax.text(pct + 1, bar.get_y() + bar.get_height()/2, 
                   f'{pct:.1f}%', 
                   va='center', fontsize=8)
        
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    summary_path = os.path.join(OUTPUT_DIR, "final_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Final summary plot saved: {summary_path}")
    print(f"\nFinal summary plot saved: {summary_path}")
    
    summary_csv_path = os.path.join(OUTPUT_DIR, "final_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info(f"Final summary CSV saved: {summary_csv_path}")

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    logger.info("Starting experiment - DATA COLLECTION PHASE")
    
    if USE_OLLAMA:
        try:
            test_response = requests.get("http://localhost:11434/api/tags", timeout=5)
            test_response.raise_for_status()
            logger.info("Ollama is running and ready")
        except Exception as e:
            logger.error(f"Ollama is not available: {str(e)}")
            logger.error("Please start Ollama: ollama serve")
            exit(1)
    
    all_results = {}
    all_full_responses = {}

    for model_name in MODELS:
        try:
            res, full_resps = run_model(model_name)
            all_results[model_name] = res
            all_full_responses[model_name] = full_resps
            if res is not None:
                save_raw_csv(model_name, res, full_resps)
        except Exception as e:
            logger.error(f"Failed to process model {model_name}: {str(e)}")
            all_results[model_name] = None
            all_full_responses[model_name] = None
            continue

    logger.info("Data collection completed. Please fill in 'Answer_Cat' in outputs/*.csv and then run the analysis script.")