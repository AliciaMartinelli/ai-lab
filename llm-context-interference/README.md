## LLM Context Interference Mapping

This folder contains the code, prompts, and generated plots for the experiment **“LLM Context Interference Mapping”**.

The core idea: feed several open-source language models with controlled prompt pairs \((A, B)\), and measure how their **hidden states** and **output distributions** change when multiple contexts are present at the same time. The goal is to make **context interference** visible and comparable across models.

---

## High-level idea

Large language models are typically prompted with a lot of context: documents, system prompts, tool outputs, previous turns, and more. We often assume that the model “mixes” this context in a reasonable way.

This experiment systematically probes:

- how much the model’s output distribution changes when two contexts \(A\) and \(B\) are present together, and  
- how this interference depends on the **type of context pair** and on the **model family**.

Instead of only looking at sample outputs, the experiment focuses on:

- the **geometry of the last hidden states**, and  
- the **token logit / probability distributions** over a fixed candidate set.

This allows us to detect **non-linear composition effects** that are not obvious from a few example generations.

---

## Setup

### Models

The experiment currently evaluates three open models:

- **Mistral-7B-Instruct** (`mistralai/Mistral-7B-Instruct-v0.2`)
- **Qwen2-7B-Instruct** (`Qwen/Qwen2-7B-Instruct`)
- **Phi-3 Mini** (`microsoft/Phi-3-mini-4k-instruct`)

You can adjust the list in `hidden_exp_multi.py` by editing the `MODELS` constant.

### Prompt pairs and categories

Prompt pairs \((A, B)\) are grouped into four categories:

- **complementary** – contexts that support each other
- **conflict** – contexts that contradict each other
- **control\_AeqB** – control condition where A and B are (approximately) the same
- **irrelevant** – additional context that should ideally not matter

The raw prompts live in:

- `prompts/data_complementary.txt`
- `prompts/data_conflict.txt`
- `prompts/data_irrelevant.txt`
- `prompts/data_control_AeqB.txt`

Each line has one of the following formats:

- `context_a ||| context_b ||| question ||| answer_prefix ||| candidates`
- `domain ||| context_a ||| context_b ||| question ||| answer_prefix ||| candidates`

Candidates are a comma-separated list of answer options; the script evaluates the model’s next-token probabilities restricted to this candidate set.

---

## Methodology

For each model, category, and prompt pair, the script:

1. Builds prompts for:
   - base question without extra context (\(P_0\))
   - with only context A (\(P_A\))
   - with only context B (\(P_B\))
   - with combined context A then B (\(P_{AB}\))
   - with combined context B then A (\(P_{BA}\))
2. Computes **log probabilities** for each candidate continuation under these prompts.
3. Derives baseline distributions, e.g.:
   - a “linear” baseline combining log probabilities from A and B
   - a simple mix of A and B
4. Extracts **hidden states** from a chosen transformer layer (by default: last layer, last token) for all prompts.
5. Computes:
   - **KL divergence** and **L2 distances** between the actual combined distribution \(p_{AB}\) and the baselines
   - **cosine similarities** between hidden states \(h_A, h_B, h_{AB}, h_{BA}\) and a linear reference \(h_{\text{lin}}\)
6. Aggregates statistics:
   - per prompt pair
   - per category and model
   - across models

The entire pipeline is implemented in `hidden_exp_multi.py`.

---

## Metrics and visualizations

The script generates several types of plots (PNG files) in `results_<model_id>/`:

- **Per-experiment candidate plots**  
  Bar plots comparing:
  - the baseline distribution over candidates, and  
  - the actual combined distribution \(p_{AB}\)

- **“Baseline vs. combined” curves**  
  For each experiment, the candidates are sorted by baseline probability, and two normalized curves are plotted:
  - `Baseline (p_lin, normalized)`
  - `Combined (p_AB, normalized)`

- **Box plots per category and metric**  
  For each model and category, box plots summarize:
  - \( \text{L2} \) distances and KL divergences vs. the linear baseline
  - optionally the same vs. a mixed baseline

- **Geometric metrics**  
  Box plots showing, for each category:
  - cosine similarities such as \(\cos(h_A, h_B)\), \(\cos(h_{AB}, h_A)\), \(\cos(h_{AB}, h_B)\), and \(\cos(h_{AB}, h_{\text{lin}})\)

- **Scatter plots**  
  For example:
  - KL divergence vs. blend score \(\cos(d_{AB}, d_A + d_B)\)
  - KL divergence vs. \(\cos(h_A, h_B)\)

Additionally, at the root of this folder, there are model comparison plots:

- `compare_models_KL_lin.png` – mean KL divergence per category and model  
- `compare_models_L2_lin.png` – mean L2 distance per category and model

---

## Interpreting the experiment (short version)

Some qualitative patterns observed in the current setup:

- **Mistral-7B-Instruct**
  - Shows a clear and interpretable interference profile.
  - Conflict prompts tend to produce larger KL/L2 distances than complementary or irrelevant prompts.
  - Control prompts (A ≈ B) stay close to the baseline, as desired.

- **Phi-3 Mini**
  - Behaves almost linearly in this configuration.
  - Combined states \(AB\) lie close to both A and B and to the linear reference.
  - Context interference is relatively mild; contexts are “gently” mixed.

- **Qwen2-7B-Instruct**
  - Shows unusually high KL/L2 values, even for some control prompts.
  - Geometric metrics are at times degenerate in this setup.
  - These results are treated as **exploratory and preliminary**, as they likely reflect a mismatch in scaling / configuration rather than a stable interference profile.

Across models, even **irrelevant** additional context is not always neutral: in some cases, it shifts the distribution more than expected, reminding us that “just adding more context” can have measurable side effects.

Hidden-state cosine similarities are overall high (as expected in late layers), yet **small directional shifts** can be systematically associated with meaningful changes in the output distribution. This suggests that even fine-grained geometric differences in the last layer can encode semantically relevant interference.

---

## How to run

From the repository root:

```bash
cd llm-context-interference
python hidden_exp_multi.py
```

What happens:

- prompt files are loaded from `prompts/`
- each model in `MODELS` is evaluated on all experiments
- metrics and plots are written into `results_<model_id>/`
- cross-model comparison plots are written into this folder

Be aware that:

- loading multiple 7B-class models can be resource-intensive
- on CPU or `mps`, the experiment may take a while

You can reduce the load by:

- commenting out some models in `MODELS`
- restricting the dataset (e.g. fewer lines in the prompt files)

---

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- NumPy
- Matplotlib
- Mistral-7B-Instruct, Qwen2-7B-Instruct, Phi-3 Mini

---

## Extending the experiment

Ideas for extensions:

- add new prompt categories (e.g. stylistic conflicts, hierarchical instructions, multi-step chains)
- analyse earlier transformer layers to see where interference first becomes visible
- perform a more systematic analysis of order effects (AB vs. BA) across models
- derive practical heuristics for prompt design and agent orchestration (e.g. context splitting, sequencing vs. parallel instructions)

Contributions and discussion are welcome in the main repository:  
`https://github.com/AliciaMartinelli/ai-lab`


