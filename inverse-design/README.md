# Inverse Design

---

## Introduction

Designing new electrolytes is a key challenge in battery research, where safety, performance, and longevity depend heavily on molecular properties. Traditional molecular ML models predict properties from known structures, but many real-world tasks—like discovering novel materials—require the inverse: generating molecules with desired properties.

Large Language Models (LLMs), though built for text, can be repurposed for molecular design by expressing molecules as linear notations like SMILES or p-SMILES. This enables conditional generation of molecules guided by target properties, offering a flexible approach to inverse design.

In this project, we explore inverse design for polymer electrolytes using both open-weight and API-based LLMs. We reproduce the PolyGen benchmark (minGPT), compare with GPT-4o and LLaMA-3.2-3B, and further fine-tune LLaMA on polymer SMILES using LoRA. We evaluate the generated molecules across six molecular-level metrics, and analyze their functional groups to assess alignment with conductivity prompts. Our findings highlight the potential—and current limitations—of using LLMs for conditional molecular generation in materials science.

---

## Method

In this project, we explore conditional inverse design of polymer electrolytes using large language models. We start by reproducing the PolyGen minGPT results as a baseline, including its six evaluation metrics (uniqueness, novelty, validity, synthesizability, similarity, diversity). 

We then apply GPT-4o and LLaMA-3.2-3B for inverse design via natural language prompting, generating new polymer candidates under “high” and “low” conductivity conditions. 

To further align with PolyGen's conditional generation setup, we fine-tune LLaMA on the HTP-MD dataset, following PolyGen’s tokenization and conditioning strategy. The fine-tuning uses LoRA adapters to reduce trainable parameters.

Finally, we evaluate all four models (minGPT, GPT-4o, base LLaMA, fine-tuned LLaMA) using the same six metrics. Additionally, we perform functional group analysis to understand the chemical patterns favored by each model and condition.

#### Key Methods: Inverse molecular design with LLMs, LoRA fine-tuning, tokenizatio (p-SMILES), conditional generation

---

## Data

We represent molecular structures using SMILES (Simplified Molecular Input Line Entry System), a notation that encodes molecules as character sequences describing atoms and bonds.

- **What it is**: SMILES is a linear text-based format for representing chemical structures.
- **Why we use it**: SMILES strings are directly compatible with transformer-based Large Language Models (LLMs), which are designed to process sequential token data.
- **Our purpose**: By treating SMILES as a form of “chemical language,” we can leverage LLMs for molecule generation and property-conditioned design.

For training and evaluation, we use [htp_md.csv](data/raw/htp_md.csv), a dataset containing polymer candidates with SMILES representations and binary conductivity labels (high/low). These SMILES are used both as LLM input/output and for downstream property evaluation.

![SMILES representation example](images/smiles_example.jpg)

*Example of how a 2D molecular structure maps to its SMILES string.*


---

## How to run

This section outlines the key steps to reproduce our experiments and evaluate inverse design results.

### 1. Set up the environment

Install required packages:

```bash
pip install -r requirements.txt
```

Make sure your environment includes:
- `transformers`, `peft`, `datasets`, `scikit-learn`, `rdkit`, etc.
- GPU is recommended for LLM inference and fine-tuning

---

### 2. Reproduce minGPT results

Use the original PolyGen-style minGPT pipeline:

```python
notebooks/generation/minGPT_pipeline.ipynb
```

Results are saved in:

```
data/generated/
```

---

### 3. Generate SMILES using GPT-4o and base LLaMA

Prompt-based inverse design using language models:

```python
notebooks/generation/gpt4o_generate.ipynb
notebooks/generation/llama_generate.ipynb
```

Cleaned outputs are stored under:

```
data/generated/
```

---

### 4. Fine-tune LLaMA on HTP-MD (PolyGen-style)

Fine-tune `meta-llama/Llama-3.2-3B-Instruct` using LoRA on HTP-MD data:

```python
notebooks/generation/llama_finetune_htpmd.ipynb
```

Input format:

```
[<HIGH>, <HIGH>, <HIGH>, <HIGH>, <HIGH>] + SMILES
```

The fine-tuned model is saved under:

```
models/llama-polygen-htpmd-lora/
```

---

### 5. Generate with fine-tuned LLaMA

Conditionally generate p-SMILES using the fine-tuned model:

```python
notebooks/generation/llama_generate_tuned.ipynb
```

---

### 6. Evaluate generated SMILES

Compute all evaluation metrics:

- Validity
- Uniqueness
- Novelty
- Synthesizability
- Similarity
- Diversity

Also includes functional group analysis.

Run:

```python
notebooks/evaluation/compute_metrics.ipynb
```

Outputs are saved to:

```
results/metrics/
```

---

## Project structure

The repository is organized as follows:

    inverse-design
    ├── data
    │   ├── generated    # SMILES generated by different models
    │   └── raw          # Source datasets, including HTP-MD data and PolyGen training splits
    ├── minGPT           # Reference or local copy of the original minGPT implementation
    ├── models
    │   └── llama-polygen-htpmd-lora    # LoRA adapter and tokenizer for the fine-tuned LLaMA model
    ├── notebooks
    │   ├── baseline     # Baseline experiments 
    │   ├── evaluation   # Evaluation metrics and functional group analysis
    │   └── generation   # SMILES generation using different models
    ├── requirements.txt
    └── results
        └── metrics      # Aggregated evaluation outputs

---

## Acknowledgements 

This repository is based on the open-source project **PolyGen** developed by the TRI-AMDD team:

- Original repository: https://github.com/TRI-AMDD/PolyGen

Parts of the code and experimental setup in this repository were adapted from PolyGen.  
In particular, the original PolyGen implementation incorporates and builds upon prior work on:

- minGPT by Karpathy: https://github.com/karpathy/minGPT  

This fork reproduces selected components of PolyGen and extends them with additional experiments and analyses.  
We welcome questions, suggestions, and contributions from the community.

---

## Reference

@article{yang2023novo, title={De novo design of polymer electrolytes with high conductivity using gpt-based and diffusion-based generative models}, author={Yang, Zhenze and Ye, Weike and Lei, Xiangyun and Schweigert, Daniel and Kwon, Ha-Kyung and Khajeh, Arash}, journal={arXiv preprint arXiv:2312.06470}, year={2023} }