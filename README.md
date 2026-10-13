<div align="center">

  <h1>ml-prompt-engineering</h1>
  <p><strong>LLM Prompt Engineering, Inference Pipelines & Fine-Tuning Convergence Telemetry</strong></p>

  <p>
    <img src="https://img.shields.io/badge/Language-Python_3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/Framework-HuggingFace_Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="HuggingFace" />
    <img src="https://img.shields.io/badge/Backend-PyTorch_2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
    <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="License" />
  </p>

</div>

---

### Overview

`ml-prompt-engineering` provides modular Python tooling for causal language model inference, prompt parameter tuning (temperature, nucleus sampling), and visualization of LLM training convergence metrics (Cross-Entropy Loss vs. Perplexity).

Key features:
- **Accelerated Ingestion:** Automatic hardware acceleration detection (`cuda`, `mps`, `cpu`).
- **Configurable Sampling:** Explicit control over `temperature`, `top_p`, and maximum token length.
- **Evaluation Telemetry:** Automatic computation and plotting of model perplexity alongside loss curves ($PPL = e^{Loss}$).

---

### Repository Layout

```text
ml-prompt-engineering/
├── src/
│   └── prompts.py        # Core causal LM inference and generation wrapper
├── test_prompts.py       # Loss and perplexity visualization evaluator
├── requirements.txt      # Pinned production dependencies
└── README.md
```

---

### Quickstart

#### 1. Environment Setup

```bash
# Create and activate a Python virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Model Inference

```python
from src.prompts import generate_completion

prompt = "Explain the difference between Kubernetes and Docker in simple terms:"
response = generate_completion(
    prompt=prompt,
    model_name="gpt2",
    max_new_tokens=80,
    temperature=0.7
)

print(response)
```

#### 3. Metric Plotting & Convergence Analysis

Run the evaluation script to visualize loss curves and perplexity trends:

```bash
python test_prompts.py
```

---

### License

Distributed under the MIT License. Developed and maintained by [Brandon Mendieta](https://github.com/NeoScraids).
