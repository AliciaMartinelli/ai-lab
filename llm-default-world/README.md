## Model Priors & Average World Experiment

This folder contains the code and generated results for the experiment **"Model Priors & Average World"**.

The core idea: repeatedly ask a language model minimal, unconstrained questions about human attributes, and measure its **unguided one-word priors**. This reveals the statistical "average human" and "average world" encoded in the model's training data.

---

## High-level idea

Language models approximate a probability distribution over human language:

\[ P_\theta(x) \approx P_{\text{human}}(x) \]

By repeatedly sampling minimal prompts like:

> "What gender does a typical human being have?"

we directly estimate:

\[ P_\theta(\text{attribute}) \]

This yields a **quantitative measurement of the model's implicit cultural and demographic defaults**, including gender, skin color, religion, age, residence type, native language, profession, sexual orientation, education level, economic status, political orientation, family status, and health status.

The experiment does not measure real-world population statistics. Instead, it measures what the model "thinks" is most likely when given no additional context—the maximum-likelihood default world encoded in its weights.

---

## Setup

### Models

The experiment currently evaluates five models via Ollama:

- **Mistral-7B-Instruct** (`mistral:7b-instruct`)
- **Llama 3 8B** (`llama3:8b`)
- **Qwen2 7B** (`qwen2:7b`)
- **Phi-3** (`phi3:latest`)
- **Gemma 2 9B** (`gemma2:9b`)

You can adjust the list in `model_priors.py` by editing the `MODELS` constant.

**Prerequisites**: All models must be installed and available via Ollama. Install models using:
```bash
ollama pull mistral:7b-instruct
ollama pull llama3:8b
ollama pull qwen2:7b
ollama pull phi3:latest
ollama pull gemma2:9b
```

### Attributes measured

Each attribute is measured without predefined categories. The prompts are:

- **Gender**: "What gender does a typical human being have?"
- **Skin color**: "What skin color does a typical human being have?"
- **Religion**: "What religion does a typical human being belong to?"
- **Age**: "How old is a typical human?"
- **Residence**: "Where does a typical human being typically live?"
- **Language**: "What is a typical human being's native language?"
- **Profession**: "What is a typical human being's typical profession?"
- **Sexual orientation**: "What is a typical human being's sexual orientation?"
- **Education level**: "What level of education does a typical human being have?"
- **Economic status**: "What is a typical human being's economic status?"
- **Political orientation**: "What is a typical human being's political orientation?"
- **Family status**: "What is a typical human being's family status?"
- **Health status**: "What is a typical human being's health status?"

All prompts are structured to request JSON responses with a single "answer" key. The script extracts and stores the complete model response for manual categorization.

---

## Methodology

The experiment follows a two-stage workflow:

### Stage 1: Data Collection (`model_priors.py`)

For each model and attribute, the script:

1. Generates \( N = 100 \) responses using JSON-formatted prompts.
2. Stores the complete model output (not just the first word) in CSV files.
3. Saves raw data to `outputs/{model_name}.csv` with columns:
   - `Sample`: Sample number (1-100)
   - `Attribute`: The attribute being measured
   - `Output`: Complete raw model response
   - `Extracted_Answer`: Parsed answer from JSON (may need manual correction)
   - `Answer_Cat`: Empty column for manual categorization
   - `Prompt`: The full prompt used
   - `Model`: Model name

The script uses Ollama API for fast inference, with temperature \( T = 0.8 \) and top-p \( = 0.95 \).

### Stage 2: Manual Categorization

1. Open each CSV file in `outputs/`.
2. Review the `Extracted_Answer` column and manually fill in the `Answer_Cat` column with standardized categories.
3. This step is crucial for accurate analysis, as automated parsing may miss nuances or fail for complex responses (especially for numeric attributes like age).

### Stage 3: Analysis (`analyze_categorized.py`)

After manual categorization, run the analysis script:

1. Reads all categorized CSV files from `outputs/`.
2. Groups responses by attribute and model.
3. Computes frequency counts and proportions for each category.
4. Calculates 95% confidence intervals using the normal approximation:
   \[ \hat{p} = \frac{k}{N}, \quad SE = \sqrt{\frac{\hat{p}(1 - \hat{p})}{N}}, \quad CI_{95} = \hat{p} \pm 1.96 \cdot SE \]
5. Generates individual plots per model and attribute.
6. Creates summary CSV files with aggregated statistics.
7. Generates a final summary plot showing the top category and percentage for each attribute across all models.

### Stage 4: Comparison Plots (`create_comparison_plots.py`)

Creates side-by-side comparison plots for each attribute across all models:

1. Reads all `*_summary.csv` files from `results_categorized/`.
2. For each attribute, creates a plot showing all 5 models side-by-side.
3. Saves plots as `comparison_{attribute}.png` in `results_categorized/`.

---

## Metrics and visualizations

The scripts generate several types of outputs:

### Data Collection (`model_priors.py`)

- **Raw CSV files**  
  One file per model in `outputs/`: `{model_name}.csv`  
  Contains all individual responses with full model output for manual review.

- **Log file**  
  `results/experiment.log` contains detailed logging of all operations, errors, and warnings.

### Analysis (`analyze_categorized.py`)

- **Summary CSV files**  
  One file per model in `results_categorized/`: `{model_name}_summary.csv`  
  Columns: `model, attribute, value, count, p, ci_low, ci_high`

- **Individual plots**  
  For each model and attribute: `{model_name}_{attribute}.png`  
  Shows percentages with confidence intervals

- **Final summary**  
  `results_categorized/final_summary.csv` and `final_summary.png`  
  Shows the top category and percentage for each attribute across all models

### Comparison Plots (`create_comparison_plots.py`)

- **Comparison plots**  
  For each attribute: `results_categorized/comparison_{attribute}.png`  
  Side-by-side comparison of all 5 models using percentages

---

## Interpreting the experiment

Some qualitative patterns to look for:

- **Model-specific differences**  
  Different models may show different priors due to training data, architecture, or alignment. For example, models trained on different language distributions may show different default languages.

- **Attribute-specific patterns**  
  Some attributes (e.g., profession) may show more diversity than others (e.g., gender). The entropy and distribution shape can reveal how "default" the model's assumptions are.

- **Cultural and demographic biases**  
  The most frequent answers reveal what the model considers "typical" or "default" for each attribute. This reflects biases encoded in the training data.

- **Confidence intervals**  
  The width of confidence intervals indicates statistical uncertainty. Narrow intervals for high-frequency answers suggest stable priors.

The experiment makes implicit assumptions explicit and measurable. A model that consistently answers "male" for gender, "white" for skin color, or "English" for language is encoding specific cultural defaults that may not match global diversity.

---

## How to run

### Prerequisites

1. Install and start Ollama:
   ```bash
   # Install Ollama (if not already installed)
   # macOS: brew install ollama
   # Or download from https://ollama.ai
   
   # Start Ollama service
   ollama serve
   ```

2. Pull required models (see Setup section above).

3. Activate your Python environment and install dependencies:
   ```bash
   pip install pandas matplotlib scipy requests tqdm
   ```

### Step 1: Data Collection

From the `llm-default-world` directory:

```bash
python model_priors.py
```

**What happens:**
- Each model in `MODELS` is queried via Ollama API
- For each attribute, \( N = 100 \) samples are generated
- Raw CSV files are written to `outputs/`
- All operations are logged to `results/experiment.log`

**Expected duration:** With Ollama, this should take approximately 10-20 minutes for all 5 models and 13 attributes (6500 total samples).

### Step 2: Manual Categorization

1. Open each CSV file in `outputs/` (e.g., `mistral_7b-instruct.csv`).
2. Review the `Extracted_Answer` column.
3. Fill in the `Answer_Cat` column with standardized categories.
   - Use consistent category names across all models for the same attribute.
   - For numeric attributes (age), you may need to create bins (e.g., "20-30", "30-40").
   - Save the files after categorization.

### Step 3: Analysis

After categorizing all CSV files:

```bash
python analyze_categorized.py
```

**What happens:**
- Reads all categorized CSVs from `outputs/`
- Generates summary CSVs and plots in `results_categorized/`
- Creates final summary plot

### Step 4: Comparison Plots (Optional)

To create side-by-side comparison plots:

```bash
python create_comparison_plots.py
```

**What happens:**
- Reads all `*_summary.csv` files from `results_categorized/`
- Creates comparison plots for each attribute
- Saves plots to `results_categorized/`

---

## Technologies

- **Python 3.12+**
- **Ollama** - Fast local LLM inference (optimized for Apple Silicon)
- **Pandas** - CSV handling and data manipulation
- **Matplotlib** - Plotting and visualization
- **SciPy** - Statistical calculations (confidence intervals)
- **Requests** - HTTP client for Ollama API

**Models tested:**
- Mistral-7B-Instruct
- Llama 3 8B
- Qwen2 7B
- Phi-3
- Gemma 2 9B

---

## Extending the experiment

Ideas for extensions:

- **More attributes**: Add additional demographic or cultural attributes
- **Conditioned priors**: Measure how priors change when given additional context (e.g., "There is a human being in Africa. What language...")
- **Cross-language experiments**: Run the same prompts in different languages and compare priors
- **Additional models**: Test other model families or sizes
- **Distribution comparison metrics**: Compute KL divergence or other metrics between models' priors
- **Temporal analysis**: Track how priors change across model versions or training checkpoints
- **Diversity metrics**: Compute entropy, Simpson index, or other diversity measures for each attribute
- **Automated categorization**: Develop NLP-based methods to automatically categorize responses (with human validation)

Contributions and discussion are welcome in the main repository:  
`https://github.com/AliciaMartinelli/ai-lab`
