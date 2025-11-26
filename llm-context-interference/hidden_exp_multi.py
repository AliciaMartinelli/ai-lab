import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
import textwrap
from collections import defaultdict

# ------------------------------------------------------
# Input text files
# Format: context_a ||| context_b ||| question ||| answer_prefix ||| candidates
# Files are expected in the local "prompts" subfolder next to this script.
# ------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")

CATEGORY_FILES = {
    "complementary": os.path.join(PROMPTS_DIR, "data_complementary.txt"),
    "conflict":      os.path.join(PROMPTS_DIR, "data_conflict.txt"),
    "irrelevant":    os.path.join(PROMPTS_DIR, "data_irrelevant.txt"),
    "control_AeqB":  os.path.join(PROMPTS_DIR, "data_control_AeqB.txt"),
}

# ------------------------------------------------------
# List of models to evaluate.
# ------------------------------------------------------
MODELS = [
    ("mistral7b", "mistralai/Mistral-7B-Instruct-v0.2"),
    ("qwen2_7b",  "Qwen/Qwen2-7B-Instruct"),
    ("phi3_mini", "microsoft/Phi-3-mini-4k-instruct"),
]

# Hidden layer index
HIDDEN_LAYER_INDEX = -1

# Device selection
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")


# ======================================================
# 1. Load experiments
# ======================================================

def load_experiments_from_files():
    """
    Supports two formats:
    5 fields:
      context_a ||| context_b ||| question ||| answer_prefix ||| candidates

    6 fields:
      domain ||| context_a ||| context_b ||| question ||| answer_prefix ||| candidates
    """
    experiments = []
    for cat, path in CATEGORY_FILES.items():
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split("|||")]

                if len(parts) == 6:
                    domain, context_a, context_b, question, answer_prefix, cand_str = parts
                elif len(parts) == 5:
                    domain = "color"  # Default if no domain column is provided
                    context_a, context_b, question, answer_prefix, cand_str = parts
                else:
                    print(f"[WARN] {path}:{line_no} has {len(parts)} fields, expected 5 or 6.")
                    continue

                candidates = [c.strip() for c in cand_str.split(",") if c.strip()]
                name = f"{domain}_{cat}_line{line_no}"

                experiments.append({
                    "name":          name,
                    "category":      cat,
                    "domain":        domain,
                    "context_a":     context_a,
                    "context_b":     context_b,
                    "question":      question,
                    "answer_prefix": answer_prefix,
                    "candidates":    candidates,
                })
    return experiments


def build_prompt(context: str, question: str, answer_prefix: str) -> str:
    ctx = context.strip()
    if ctx:
        return f"Context:\n{ctx}\n\nQuestion:\n{question}\n\n{answer_prefix} "
    else:
        return f"Question:\n{question}\n\n{answer_prefix} "


# ======================================================
# 2. Model helper functions
# ======================================================

def candidate_logprobs_for_prompt(prompt: str, candidates, tokenizer, model, device) -> np.ndarray:
    """
    Compute the sum of log probabilities for the token sequence of each candidate.
    """
    base_enc = tokenizer(prompt, return_tensors="pt")
    base_ids = base_enc["input_ids"].to(device)
    base_len = base_ids.shape[1]

    logps = []
    for w in candidates:
        full_text = prompt + w
        enc = tokenizer(full_text, return_tensors="pt")
        ids = enc["input_ids"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=ids, use_cache=False)
        logits = outputs.logits[0]          # [seq_len, vocab]
        log_probs = torch.log_softmax(logits, dim=-1)

        seq_len = ids.shape[1]
        cand_start = base_len
        cand_end   = seq_len

        lp_sum = 0.0
        for pos in range(cand_start, cand_end):
            token_id = ids[0, pos]
            lp_sum += log_probs[pos - 1, token_id].item()
        logps.append(lp_sum)

    return np.array(logps, dtype=np.float64)


def get_hidden_for_prompt(prompt: str, tokenizer, model, device, layer_index: int = -1) -> np.ndarray:
    """
    Use the last token of the specified layer as the representation.
    """
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states
    h_layer = hidden_states[layer_index][0]   # [seq, dim]
    last_token = h_layer[-1]                 # [dim]
    return last_token.detach().cpu().numpy()


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (ex.sum() + 1e-12)


def cos_sim(u, v):
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)

    if not np.isfinite(nu) or not np.isfinite(nv):
        return 0.0

    if nu < 1e-9 or nv < 1e-9:
        return 0.0

    return float(np.dot(u, v) / (nu * nv))



# ======================================================
# 3. Core experiment for a single model
# ======================================================

def run_for_model(model_id: str, model_name: str):
    print(f"\n\n=== Model {model_id}: {model_name} ===")
    out_dir = f"results_{model_id}"
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32,
    )
    model.to(DEVICE)
    model.eval()

    experiments = load_experiments_from_files()
    print(f"Loaded experiments: {len(experiments)}")

    metrics_per_experiment = []
    geom_metrics_per_experiment = []
    per_word_interference = defaultdict(list)
    category_patterns = defaultdict(list)   # key: (category, order)

    for exp in experiments:
        name          = exp["name"]
        cat           = exp["category"]
        dom           = exp["domain"]
        ctx_a         = exp["context_a"]
        ctx_b         = exp["context_b"]
        question      = exp["question"]
        answer_prefix = exp["answer_prefix"]
        candidates    = exp["candidates"]

        print("\n" + "=" * 100)
        print(f"Experiment: {name}  (domain={dom}, category={cat})")
        print("=" * 100)

        # Prompts
        P0  = build_prompt("",            question, answer_prefix)
        PA  = build_prompt(ctx_a,         question, answer_prefix)
        PB  = build_prompt(ctx_b,         question, answer_prefix)
        PAB = build_prompt(ctx_a + "\n" + ctx_b, question, answer_prefix)  # A then B
        PBA = build_prompt(ctx_b + "\n" + ctx_a, question, answer_prefix)  # B then A

        print("\nP0:\n",  textwrap.fill(P0, 80))
        print("\nPA:\n",  textwrap.fill(PA, 80))
        print("\nPB:\n",  textwrap.fill(PB, 80))
        print("\nPAB (A then B):\n", textwrap.fill(PAB, 80))
        print("\nPBA (B then A):\n", textwrap.fill(PBA, 80))

        # Log probabilities
        logP0  = candidate_logprobs_for_prompt(P0,  candidates, tokenizer, model, DEVICE)
        logPA  = candidate_logprobs_for_prompt(PA,  candidates, tokenizer, model, DEVICE)
        logPB  = candidate_logprobs_for_prompt(PB,  candidates, tokenizer, model, DEVICE)
        logPAB = candidate_logprobs_for_prompt(PAB, candidates, tokenizer, model, DEVICE)
        logPBA = candidate_logprobs_for_prompt(PBA, candidates, tokenizer, model, DEVICE)

        # Baselines (identical for both orders)
        logP_lin = logPA + logPB - logP0
        logP_mix = 0.5 * (logPA + logPB)

        # Hidden states
        h0  = get_hidden_for_prompt(P0,  tokenizer, model, DEVICE, HIDDEN_LAYER_INDEX)
        hA  = get_hidden_for_prompt(PA,  tokenizer, model, DEVICE, HIDDEN_LAYER_INDEX)
        hB  = get_hidden_for_prompt(PB,  tokenizer, model, DEVICE, HIDDEN_LAYER_INDEX)
        hAB = get_hidden_for_prompt(PAB, tokenizer, model, DEVICE, HIDDEN_LAYER_INDEX)
        hBA = get_hidden_for_prompt(PBA, tokenizer, model, DEVICE, HIDDEN_LAYER_INDEX)
        
        for label, h in [("h0", h0), ("hA", hA), ("hB", hB), ("hAB", hAB), ("hBA", hBA)]:
            if not np.all(np.isfinite(h)):
                print(f"[WARN] Non-finite hidden state in {label} for experiment {name}")
                
        # Two orders: AB and BA
        for order_label, logP_combo, hAB_curr in [
            ("AB", logPAB, hAB),
            ("BA", logPBA, hBA),
        ]:
            p_lin = softmax_np(logP_lin)
            p_mix = softmax_np(logP_mix)
            p_AB  = softmax_np(logP_combo)

            delta_lin = p_AB - p_lin
            delta_mix = p_AB - p_mix

            L2_lin = float(np.linalg.norm(delta_lin, ord=2))
            L2_mix = float(np.linalg.norm(delta_mix, ord=2))
            eps = 1e-9
            KL_lin = float(np.sum(p_AB * (np.log(p_AB + eps) - np.log(p_lin + eps))))
            KL_mix = float(np.sum(p_AB * (np.log(p_AB + eps) - np.log(p_mix + eps))))

            print(f"\n[{order_label}] Interference metrics ({len(candidates)} candidates):")
            print(f"  [lin] L2   = {L2_lin:.4f}, KL = {KL_lin:.4f}")
            print(f"  [mix] L2   = {L2_mix:.4f}, KL = {KL_mix:.4f}")

            metrics_per_experiment.append({
                "model":          model_id,
                "name":           name,
                "order":          order_label,
                "category":       cat,
                "domain":         dom,
                "L2_lin":         L2_lin,
                "KL_lin":         KL_lin,
                "L2_mix":         L2_mix,
                "KL_mix":         KL_mix,
                "num_candidates": len(candidates),
            })

            # Per-word interference
            for w, d in zip(candidates, delta_lin):
                per_word_interference[w].append({
                    "model":     model_id,
                    "experiment": name,
                    "order":     order_label,
                    "domain":    dom,
                    "category":  cat,
                    "delta":     float(d),
                })

            # Candidate bar plot
            x = np.arange(len(candidates))
            width = 0.35

            plt.figure(figsize=(8, 4))
            plt.bar(x - width/2, p_lin, width=width, label="p_lin (Baseline)")
            plt.bar(x + width/2, p_AB,  width=width, label="p_AB (actual)")
            plt.xticks(x, candidates, rotation=45, ha="right")
            plt.ylabel("Probability over candidates")
            plt.title(f"{name} [{order_label}] (model={model_id}, category={cat})")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(out_dir, f"exp_{name}_{order_label}_candidates.png")
            plt.savefig(fname, dpi=200)
            plt.close()

            order_idx    = np.argsort(-p_lin)
            p_lin_sorted = p_lin[order_idx]
            p_AB_sorted  = p_AB[order_idx]

            category_patterns[(cat, order_label)].append((p_lin_sorted, p_AB_sorted))

            p_lin_norm = p_lin_sorted / (p_lin_sorted.max() + eps)
            p_AB_norm  = p_AB_sorted  / (p_AB_sorted.max()  + eps)
            xs = np.arange(len(p_lin_sorted))

            plt.figure(figsize=(8, 3))
            plt.plot(xs, p_lin_norm, "--", label="Baseline (p_lin, normalized)")
            plt.plot(xs, p_AB_norm,  "-",  label="Combined (p_AB, normalized)")
            plt.xlabel("Candidate index (sorted by p_lin)")
            plt.ylabel("Normalized intensity")
            plt.title(f"{name} [{order_label}] Baseline vs. combined distribution")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            fname = os.path.join(out_dir, f"exp_{name}_{order_label}_baseline_vs_combined.png")
            plt.savefig(fname, dpi=200)
            plt.close()

            # Geometry in hidden space
            dA   = hA       - h0
            dB   = hB       - h0
            dAB  = hAB_curr - h0
            d_lin = dA + dB

            cos_A_B     = cos_sim(hA,       hB)
            cos_AB_A    = cos_sim(hAB_curr, hA)
            cos_AB_B    = cos_sim(hAB_curr, hB)
            cos_AB_lin  = cos_sim(hAB_curr, d_lin)
            blend_score = cos_sim(dAB,      d_lin)   # cos(dAB, dA + dB)

            geom_metrics_per_experiment.append({
                "model":       model_id,
                "name":        name,
                "order":       order_label,
                "category":    cat,
                "domain":      dom,
                "cos_A_B":     cos_A_B,
                "cos_AB_A":    cos_AB_A,
                "cos_AB_B":    cos_AB_B,
                "cos_AB_lin":  cos_AB_lin,
                "blend_score": blend_score,
            })

    # ==================================================
    # 4. Box plots per model
    # ==================================================

    if metrics_per_experiment:
        categories = sorted(set(m["category"] for m in metrics_per_experiment))

        def make_box(metric_key: str, title_suffix: str, fname_suffix: str):
            data = [
                [m[metric_key] for m in metrics_per_experiment if m["category"] == cat]
                for cat in categories
            ]
            plt.figure(figsize=(7, 4))
            plt.boxplot(data, tick_labels=categories, showmeans=True)
            plt.ylabel(metric_key)
            plt.title(f"{title_suffix} (model: {model_id})")
            plt.tight_layout()
            fname = os.path.join(out_dir, f"summary_{fname_suffix}_{model_id}.png")
            plt.savefig(fname, dpi=200)
            plt.close()
            print(f"Saved box plot for {metric_key} as: {fname}")

        make_box("L2_lin", "Interference strength (L2, linear baseline)", "L2_lin")
        make_box("KL_lin", "Interference strength (KL, linear baseline)", "KL_lin")
        make_box("L2_mix", "Interference strength (L2, mixed baseline)", "L2_mix")
        make_box("KL_mix", "Interference strength (KL, mixed baseline)", "KL_mix")

    # ==================================================
    # 5. Geometric box plots and scatter plots
    # ==================================================

    if geom_metrics_per_experiment:
        categories = sorted(set(g["category"] for g in geom_metrics_per_experiment))

        def make_geom_box(metric_key: str, title: str, fname_suffix: str):
            data = [
                [g[metric_key] for g in geom_metrics_per_experiment if g["category"] == cat]
                for cat in categories
            ]
            plt.figure(figsize=(7, 4))
            plt.boxplot(data, tick_labels=categories, showmeans=True)
            plt.ylabel(metric_key)
            plt.title(f"{title} (model: {model_id})")
            plt.tight_layout()
            fname = os.path.join(out_dir, f"geom_{fname_suffix}_{model_id}.png")
            plt.savefig(fname, dpi=200)
            plt.close()
            print(f"Saved geometric box plot for {metric_key} as: {fname}")

        make_geom_box(
            "blend_score",
            "Blend score cos(dAB, dA + dB) by category",
            "blend_score",
        )
        make_geom_box(
            "cos_A_B",
            "cos(h_A, h_B) (context similarity) by category",
            "cos_A_B",
        )
        make_geom_box(
            "cos_AB_A",
            "cos(h_AB, h_A) by category",
            "cos_AB_A",
        )
        make_geom_box(
            "cos_AB_B",
            "cos(h_AB, h_B) by category",
            "cos_AB_B",
        )
        make_geom_box(
            "cos_AB_lin",
            "cos(h_AB, h_lin) by category",
            "cos_AB_lin",
        )

        # Combined list for scatter plots
        combined = [
            {**m, **g}
            for m, g in zip(metrics_per_experiment, geom_metrics_per_experiment)
        ]

        colors = {
            "complementary": "tab:blue",
            "conflict":      "tab:red",
            "control_AeqB":  "tab:green",
            "irrelevant":    "tab:gray",
        }

        # Scatter: KL_lin vs blend_score
        plt.figure(figsize=(7, 5))
        seen = set()
        for row in combined:
            cat = row["category"]
            col = colors.get(cat, "black")
            label = cat if cat not in seen else None
            plt.scatter(row["blend_score"], row["KL_lin"],
                        color=col, alpha=0.7, label=label)
            seen.add(cat)
        plt.xlabel("blend_score = cos(dAB, dA + dB)")
        plt.ylabel("KL_lin")
        plt.title(f"Interference vs. blend_score (model: {model_id})")
        plt.legend()
        plt.grid(True)
        fname = os.path.join(out_dir, f"scatter_KL_lin_vs_blend_{model_id}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()

        # Scatter: KL_lin vs cos(h_A, h_B)
        plt.figure(figsize=(7, 5))
        seen = set()
        for row in combined:
            cat = row["category"]
            col = colors.get(cat, "black")
            label = cat if cat not in seen else None
            plt.scatter(row["cos_A_B"], row["KL_lin"],
                        color=col, alpha=0.7, label=label)
            seen.add(cat)
        plt.xlabel("cos(h_A, h_B)")
        plt.ylabel("KL_lin")
        plt.title(f"Interference vs. context similarity (model: {model_id})")
        plt.legend()
        plt.grid(True)
        fname = os.path.join(out_dir, f"scatter_KL_lin_vs_cosAB_{model_id}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()

    # ==================================================
    # 6. Aggregated interference patterns by category and order
    # ==================================================

    for (cat, order_label), pattern_list in category_patterns.items():
        min_len = min(arr[0].shape[0] for arr in pattern_list)
        lin_mat = np.stack([pl[:min_len] for pl, _ in pattern_list])
        ab_mat  = np.stack([pa[:min_len] for _, pa in pattern_list])

        mean_lin = lin_mat.mean(axis=0)
        mean_ab  = ab_mat.mean(axis=0)
        std_ab   = ab_mat.std(axis=0)
        xs = np.arange(min_len)

        plt.figure(figsize=(7, 4))
        plt.plot(xs, mean_lin, "--", label="mean p_lin (single)")
        plt.plot(xs, mean_ab,  "-", label="mean p_AB (double)")
        plt.fill_between(
            xs,
            mean_ab - std_ab,
            mean_ab + std_ab,
            alpha=0.2,
            label="p_AB ± 1σ",
        )
        plt.xlabel("Candidate rank (0 = top by p_lin)")
        plt.ylabel("Probability")
        plt.title(
            f"Mean interference pattern – category: {cat}, order: {order_label}, model: {model_id}"
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fname = os.path.join(out_dir, f"pattern_{cat}_{order_label}_mean_lin_{model_id}.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        print(
            f"Saved aggregated pattern plot for (category={cat}, order={order_label}) as: {fname}"
        )

    # ==================================================
    # 7. Per-word summary (for debugging / inspection)
    # ==================================================

    print("\n" + "=" * 80)
    print(
        f"Interference over all experiments aggregated by word (linear baseline) – model {model_id}"
    )
    print("=" * 80)
    for w, entries in per_word_interference.items():
        vals = [e["delta"] for e in entries]
        mean = float(np.mean(vals))
        std  = float(np.std(vals))
        cats = sorted(set(e["category"] for e in entries))
        print(f"{w:10s}  mean_delta={mean:+.4f}  std={std:.4f}  cats={cats}")

    # Free VRAM / accelerator memory
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()

    return metrics_per_experiment, geom_metrics_per_experiment


# ======================================================
# 8. Main entry point: run all models and compare
# ======================================================

def main():
    all_model_metrics = {}

    for model_id, model_name in MODELS:
        metrics, geom = run_for_model(model_id, model_name)
        all_model_metrics[model_id] = metrics

    # Compare models: mean KL_lin and L2_lin per category
    if not all_model_metrics:
        return

    categories = sorted(
        set(m["category"]
            for metrics in all_model_metrics.values()
            for m in metrics)
    )

    metrics_to_compare = ["KL_lin", "L2_lin"]

    for metric_key in metrics_to_compare:
        plt.figure(figsize=(8, 4))
        x = np.arange(len(categories))
        total_models = len(MODELS)
        width = 0.8 / max(total_models, 1)

        for i, (model_id, metrics) in enumerate(all_model_metrics.items()):
            means = []
            for cat in categories:
                vals = [m[metric_key] for m in metrics if m["category"] == cat]
                means.append(np.mean(vals) if vals else np.nan)
            offset = (i - (total_models - 1) / 2) * width
            plt.bar(x + offset, means, width=width, label=model_id)

        plt.xticks(x, categories)
        plt.ylabel(metric_key)
        plt.title(f"Model comparison – mean {metric_key} per category")
        plt.legend()
        plt.tight_layout()
        fname = f"compare_models_{metric_key}.png"
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved model comparison for {metric_key} as: {fname}")


if __name__ == "__main__":
    main()
