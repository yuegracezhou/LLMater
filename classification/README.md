# Classification

##📌 Overview

This module focuses on molecular property classification using Large Language Models (LLMs). The task involves predicting whether a given molecule—represented as a SMILES string—exhibits a specific property. The implementation is inspired by the benchmark study: [**What can Large Language Models do in chemistry? A comprehensive benchmark on eight tasks**]([https://github.com/ChemFoundationModels/ChemLLMBench](https://doi.org/10.48550/arXiv.2305.18365)), for molecular property prediction and tries to replicate the experimental setups. It takes a SMILES input and predicts whether the molecule exhibits a particular property.

##🎯 Objective
Input: SMILES representation of a molecule
Output: Binary classification label

The goal is to evaluate how effectively LLMs can act as discriminative models for molecular property prediction using:

**In-context learning (ICL)**
Different sampling strategies
Both closed and open-source models

##⚙️ Methodology
🧠 In-Context Learning (ICL)

We use few-shot prompting, where:

1. A set of labeled molecule examples is provided as context
2. A new (test) SMILES string is appended
3. The model predicts the corresponding label

##📊 Context Construction

Two sampling strategies are used to build the in-context examples:

1. Random Sampling
    Randomly selects molecules from the dataset
    Simple baseline approach

2. Scaffold Sampling
    Selects molecules based on chemical scaffolds
    Ensures structural diversity
    More chemically meaningful context

We experiment with:

4-shot prompting
7-shot prompting

These configurations help evaluate how context size impacts performance.

##🧪 Experimental Setup

🔒 Closed Model Benchmarking

We replicate the original paper’s setup as closely as possible.

Model: **GPT-4o-mini**
Prompting Strategy: **In-Context Learning**
Sampling Methods: **Random, Scaffold**
Shots: 4 and 7

⚠️ Constraints (Free API Tier)
Limited number of prompts per day
Token usage restrictions
Reduced context size compared to original paper

##🌐 Open Model Benchmarking & Fine-Tuning

We extend the experiments using an open-source model:

Base Model: LLaMA-based (e.g., Llama-3.2-3B-Instruct)
Steps:
Run the same ICL benchmarking setup as GPT
Evaluate baseline performance
Fine-tune the model using prompt–label pairs
Re-evaluate performance

🎯 Goal:

Improve:
*Stability*
*Recall*
*Overall classification performance*

# Notebook Usage

1. Download the notebooks or clone the repository to your environment.
2. Use a Jupyter or any python environment. For best results use Google Colab Notebooks
3. Create an OpenAI API key from the OpenAI website and fill in the placeholder for the API KEY
4. The notebooks will create a zip file and can be downloaded from the UI

# Credits

Original repo: [*ChemLLMBench*](https://github.com/ChemFoundationModels/ChemLLMBench)<br>
Notebook     : [*Property_Prediction*](https://github.com/ChemFoundationModels/ChemLLMBench/blob/main/Property_Prediction.ipynb)<br>
Data         : [*Data*](https://github.com/ChemFoundationModels/ChemLLMBench/blob/main/data/property_prediction/BACE.csv)
