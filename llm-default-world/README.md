## Model Priors & Average World Experiment

This folder contains the code and generated results for the experiment **"Model Priors & Average World"**.

The core idea: repeatedly ask a language model minimal, unconstrained questions about human attributes, and measure its **unguided one-word priors**. This reveals the statistical "average human" and "average world" encoded in the model's training data.

---

## High-level idea

Language models approximate a probability distribution over human language:

\[ P_\theta(x) \approx P_{\text{human}}(x) \]

By repeatedly sampling minimal prompts like:

> "There is a human being. What gender does this person have?"

we directly estimate:

\[ P_\theta(\text{attribute}) \]

This yields a **quantitative measurement of the model's implicit cultural and demographic defaults**, including gender, skin color, religion, age, residence type, native language, and profession.

The experiment does not measure real-world population statistics. Instead, it measures what the model "thinks" is most likely when given no additional context—the maximum-likelihood default world encoded in its weights.

---

## Setup

### Models

The experiment currently evaluates two Mistral models:

- **Mistral-7B-Instruct** (`mistralai/Mistral-7B-Instruct-v0.2`)
- **Mistral-7B** (`mistralai/Mistral-7B-v0.1`)

You can adjust the list in `model_priors.py` by editing the `MODELS` constant.

### Attributes measured

Each attribute is measured without predefined categories. The prompts are:

- **Gender**: "What gender does this person have?"
- **Skin color**: "What skin color does this person have?"
- **Religion**: "What religion or worldview does this person belong to?"
- **Age**: "How old is this person?"
- **Residence**: "Where does this person typically live?"
- **Language**: "What is this person's native language?"
- **Profession**: "What is this person's typical profession?"

All answers are free one-word responses. The script extracts the first word from each generation and normalizes it (lowercase, punctuation removed).

---

## Methodology

For each model and attribute, the script:

1. Samples \( N = 300 \) one-word answers using the same minimal prompt.
2. Computes frequency counts and proportions for each unique answer.
3. Calculates 95% confidence intervals using the normal approximation:
   \[ \hat{p} = \frac{k}{N}, \quad SE = \sqrt{\frac{\hat{p}(1 - \hat{p})}{N}}, \quad CI_{95} = \hat{p} \pm 1.96 \cdot SE \]
4. Generates histogram plots showing both absolute counts and percentages.
5. Creates comparison plots across models for each attribute.

The entire pipeline is implemented in `model_priors.py`.

---

## Metrics and visualizations

The script generates several types of outputs in `results/`:

- **Raw CSV files**  
  One file per model containing all individual answers: `{model}_raw.csv`

- **Summary CSV files**  
  One file per model with aggregated statistics: `{model}_summary.csv`  
  Columns: `model, attribute, value, count, p, ci_low, ci_high`

- **Individual plots**  
  For each model and attribute: `{model}_{attribute}.png`  
  Shows both absolute counts and percentages side-by-side

- **Comparison plots**  
  For each attribute: `comparison_{attribute}.png`  
  Side-by-side comparison of all models using percentages

- **Log file**  
  `experiment.log` contains detailed logging of all operations, errors, and warnings

---

## Interpreting the experiment (short version)

Some qualitative patterns to look for:

- **Differences between base and instruct models**  
  Instruct models may show different priors due to alignment training. For example, they might be more cautious or show different default assumptions.

- **Attribute-specific patterns**  
  Some attributes (e.g., profession) may show more diversity than others (e.g., gender). The entropy and distribution shape can reveal how "default" the model's assumptions are.

- **Cultural and demographic biases**  
  The most frequent answers reveal what the model considers "typical" or "default" for each attribute. This reflects biases encoded in the training data.

- **Confidence intervals**  
  The width of confidence intervals indicates statistical uncertainty. Narrow intervals for high-frequency answers suggest stable priors.

The experiment makes implicit assumptions explicit and measurable. A model that consistently answers "male" for gender, "white" for skin color, or "English" for language is encoding specific cultural defaults that may not match global diversity.

---

## How to run

From the repository root:

cd llm-default-world
python model_priors.pyWhat happens:

- each model in `MODELS` is loaded and evaluated on all attributes
- for each attribute, \( N = 300 \) samples are generated
- raw CSV files, summary CSVs, and plots are written into `results/`
- all operations are logged to `results/experiment.log`

Be aware that:

- loading multiple 7B-class models can be resource-intensive
- on CPU or `mps`, the experiment may take a while (especially with \( N = 300 \) samples per attribute)
- the script uses temperature \( T = 0.8 \) and top-p \( = 0.95 \) for sampling

You can reduce the load by:

- commenting out some models in `MODELS`
- reducing `N_SAMPLES` in the script (though this reduces statistical power)

---

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- NumPy, SciPy (for statistical calculations)
- Pandas (for CSV handling)
- Matplotlib (for plotting)
- Mistral-7B-Instruct, Mistral-7B

---

## Extending the experiment

Ideas for extensions:

- add more attributes (e.g., education level, socioeconomic status, political orientation)
- conditioned priors: measure how priors change when given additional context (e.g., "There is a human being in Africa. What language...")
- cross-language experiments: run the same prompts in different languages and compare priors
- additional models: extend to other model families (Llama, Qwen, Phi, etc.)
- distribution comparison metrics: compute KL divergence or other metrics between models' priors
- temporal analysis: track how priors change across model versions or training checkpoints
- diversity metrics: compute entropy, Simpson index, or other diversity measures for each attribute

Contributions and discussion are welcome in the main repository:  
`https://github.com/AliciaMartinelli/ai-lab`
