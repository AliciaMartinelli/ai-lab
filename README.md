## ai-lab

A collection of small to mid-sized experiments, evaluations, and prototypes exploring how large language models change the way we build, learn, and create.

The goal of this repo is **exploration and transparency**, not production-readiness: each experiment is intended to be easy to read, re-run, and extend.

---

## Structure

- `llm-context-interference/`  
  Experiment code and data for **LLM Context Interference Mapping** (see dedicated README in that folder).
- `llm-default-world/`  
  Experiment code and data for **Model Priors & Average World** (see dedicated README in that folder).

---

## Getting started

1. **Clone the repository**

   ```bash
   git clone https://github.com/AliciaMartinelli/ai-lab.git
   cd ai-lab
   ```

2. **Install base dependencies for LLM experiments**  
   (ideally inside a virtual environment of your choice)

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install transformers matplotlib numpy pandas scipy requests tqdm
   ```

   Adjust the `torch` installation if you want a CUDA-enabled build.
   
   **Note**: The `llm-default-world` experiment requires Ollama for model inference. See the experiment's README for setup instructions.

3. **Run a specific experiment**

   See the README inside each subfolder, e.g.:

   ```bash
   cd llm-context-interference
   python hidden_exp_multi.py
   ```

---

## Experiments

### LLM Context Interference Mapping

Located in `llm-context-interference/` (with its own `README.md`).

This experiment probes how different open-source language models combine multiple pieces of context.  
Controlled prompt pairs \((A, B)\) are constructed across four categories:

- complementary
- conflicting
- control (A = B)
- irrelevant

For each model and prompt pair, the code measures:

- the geometry of the final hidden representations, and  
- the output distributions over a fixed candidate set

for A, B, and their combinations (AB, BA). It then compares these against simple linear or mixed baselines to quantify **context interference**.

For full details, metrics, and visualizations, see the dedicated README in `llm-context-interference/`.

### Model Priors & Average World

Located in `llm-default-world/` (with its own `README.md`).

This experiment measures the **unguided one-word priors** that language models assign to various human attributes. By repeatedly asking minimal, unconstrained questions like "What gender does a typical human being have?", we directly estimate the statistical "average human" and "average world" encoded in the model's training data.

The experiment evaluates five models (Mistral-7B-Instruct, Llama 3 8B, Qwen2 7B, Phi-3, Gemma 2 9B) across 13 attributes:

- gender, skin color, religion, age, residence, language, profession
- sexual orientation, education level, economic status, political orientation, family status, health status

For each model and attribute, the code:

- generates \( N = 100 \) responses using JSON-formatted prompts
- stores complete model outputs for manual categorization
- computes frequency counts, proportions, and 95% confidence intervals
- generates individual plots per model/attribute and comparison plots across models

The results reveal **quantitative measurements of implicit cultural and demographic defaults** encoded in each model, making implicit assumptions explicit and measurable.

For full details, methodology, and visualizations, see the dedicated README in `llm-default-world/`.

---

## Contributing

This repo is intentionally exploratory. If you want to add a new experiment, extend an existing one, or discuss results:

- open an issue or pull request on GitHub, or  
- reach out via the contact options linked on the project website.

Please keep code readable, experiments reproducible, and document any non-obvious design choices.
