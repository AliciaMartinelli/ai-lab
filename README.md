## ai-lab

A collection of small to mid-sized experiments, evaluations, and prototypes exploring how large language models change the way we build, learn, and create.

The goal of this repo is **exploration and transparency**, not production-readiness: each experiment is intended to be easy to read, re-run, and extend.

---

## Structure

- `llm-context-interference/`  
  Experiment code and data for **LLM Context Interference Mapping** (see dedicated README in that folder).
- More experiments will be added over time.

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
   pip install transformers matplotlib numpy
   ```

   Adjust the `torch` installation if you want a CUDA-enabled build.

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

---

## Contributing

This repo is intentionally exploratory. If you want to add a new experiment, extend an existing one, or discuss results:

- open an issue or pull request on GitHub, or  
- reach out via the contact options linked on the project website.

Please keep code readable, experiments reproducible, and document any non-obvious design choices.
