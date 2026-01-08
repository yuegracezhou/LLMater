# Classification

This approach tries to replicate the experimental setups performed in  
[ChemLLMBench](https://github.com/ChemFoundationModels/ChemLLMBench)  
for molecular property prediction. It takes a SMILES input and predicts whether the molecule exhibits a particular property.

# Method

This approach utilizes **In-Context Learning (ICL)** using **GPT-4o-mini**. Several changes have been made to the original experimental setup in order to accommodate the constraints of the free version of the OpenAI model.

We select a number of samples using **random sampling** and **scaffold sampling**, which are provided as context along with the prompt. The actual test sample is then appended, and the model is asked to predict the corresponding property.

# Closed Model Benchmarking

In this benchmarking setup, we aim to replicate the experimental conditions described in the original paper as closely as possible in order to verify the reported results.

[result](classification/Screenshot (148).png)

# Open Model Fine-Tuning

In this phase of the project, we first perform the same benchmarking experiments using an open-source **LLaMA-based model**. We then fine-tune the model to achieve improved performance on molecular property prediction tasks.
