# Classification

This approach tries to replicate the experimental setups for molecular property prediction performed in  [What can Large Language Models do in chemistry? A comprehensive benchmark on eight tasks]([https://github.com/ChemFoundationModels/ChemLLMBench](https://doi.org/10.48550/arXiv.2305.18365))  
for molecular property prediction. It takes a SMILES input and predicts whether the molecule exhibits a particular property.

# Method

This approach utilizes **In-Context Learning (ICL)** using **GPT model**. Several changes have been made to the original experimental setup in order to accommodate the constraints of the free version of the OpenAI model.

We select a number of samples using **random sampling** and **scaffold sampling**, which are provided as context along with the prompt. The actual test sample is then appended, and the model is asked to predict the corresponding property.

# Closed Model Benchmarking

In this benchmarking setup, we aim to replicate the experimental conditions described in the original paper as closely as possible in order to verify the reported results.

Model: **GPT-4o-mini**<br>
Sampling: **Random**, **Scaffold**<br>
Number of samples used: **4, 7**

Limitations of free version: Number of Prompts per day, Number of tokens per day

# Open Model Fine-Tuning

In this phase of the project, we first perform the same benchmarking experiments using an open-source **LLaMA-based model**. We then fine-tune the model to achieve improved performance on molecular property prediction tasks.

# Notebook Usage

1. Download the notebooks or clone the repository to your environment.
2. Use a Jupyter or any python environment. For best results use Google Colab Notebooks
3. Create an OpenAI API key from the OpenAI website and fill in the placeholder for the API KEY
4. The notebooks will create a zip file and can be downloaded from the UI

# Credits

Original repo: [*ChemLLMBench*](https://github.com/ChemFoundationModels/ChemLLMBench)<br>
Notebook     : [*Property_Prediction*](https://github.com/ChemFoundationModels/ChemLLMBench/blob/main/Property_Prediction.ipynb)<br>
Data         : [*Data*](https://github.com/ChemFoundationModels/ChemLLMBench/blob/main/data/property_prediction/BACE.csv)
