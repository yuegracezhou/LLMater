# LLMater 🧪 Data Science in a Molecular Universe

Supervisor: Chao Zhang (Department of Chemistry, Uppsala University)

Group Members: André Ramos Ekengren, Ramkishor Prabhu Ramlal, Yue Zhou

---

## Table of contents

- [Topic Overview](#topic-overview)

- [Literature Survey](#Literature-Survey (from 2022))

- [Methodologies](#Methodologies)

- [Benchmark Results](#Benchmark-Results)

- [Future work](#Future-work)

---

## Topic Overview

Large language models (LLMs) have shown impressive capabilities in domains such as natural language processing, code generation, and reasoning. In parallel, chemistry and materials science are becoming increasingly data-driven, with growing databases of reactions, molecules, and material properties. Bridging these two developments creates an opportunity: if molecular information can be expressed in a form that is compatible with LLMs, then models originally designed for human language might also support molecular property prediction and materials design.

In this project we focus on molecular prediction problems in material design that are relevant for battery research. Electrolytes play a central role in battery performance, safety, and lifetime, yet their design space is extremely large. Traditional machine-learning models have been successfully used for some property-prediction tasks, but they often require carefully engineered descriptors and do not easily generalize across different tasks or datasets. LLMs offer an alternative in which the same transformer architecture can be applied to a wide range of problems once the molecules are encoded in a suitable way.

---

## Literature Survey (from 2022)

As a first step, we conducted a literature survey on LLM applications in chemistry and materials science (primarily from 2022 onwards). The survey covers:

- Property prediction from molecular or materials representations, often using SMILES or graph-based encodings.
- Generative design of molecules and materials with targeted properties.
- Studies that probe the limitations of LLMs in scientific and multimodal settings.

The relevant papers are collected and summarized in a the table below.

| Model name | Architecture | Data link | Description | Adaptation | Purpose | Year | DOI | Github link |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| LLM-prop | T5 (encoder) | [Benchmark data](https://drive.google.com/drive/folders/1YCDBzwjwNRIc1FRkB662G3Y5AOWaokUG?ths=true) | predicting the properties of crystalline materials using large language models | Fine-tuning | Regression | 2025 | [DOI](https://doi.org/10.1038/s41524-025-01536-2) | [Github](https://github.com/vertaix/LLM-Prop) |
| N/A | GPT-J-6B,  Llama-3.1-8B,  Mistral-7B | [Github](https://github.com/JorenBE/GPT-Challenge) | Assessment of fine-tuned large language models for real-world chemistry and material science applications  | Fine-tuning | Classification Regression Inverse design | 2025 | [DOI](https://doi.org/10.1039/D4SC04401K) | [Github](https://github.com/JorenBE/GPT-Challenge) |
| MolRAG | Llama-3-8B-Instruct GPT-4o Qwen2.5-7B-Instruct | N/A | N/A | N/A | N/A | N/A | [DOI](https://doi.org/10.18653/v1/2025.acl-long.755) | [Github](https://github.com/AcaciaSin/MolRAG), Code not available yet. |
| ElaTBot ElaTBot-DFT | Llama2-7b, GPT-4o | [Data](https://figshare.com/articles/dataset/Large_Language_Models_for_Material_Property_Predictions_elastic_constant_tensor_prediction_and_materials_design/28399757/1?file=54029759) | Large language models for material property predictions: elastic constant tensor prediction and materials design | Fine-tuning, (LORA+), Prompt engineering, RAG  | Regression, Inverse design / Material generation | 2025 | [DOI](https://doi.org/10.1039/D5DD00061K) | [Github](https://github.com/Grenzlinie/ElaTBot) |
| N/A | LLaMA 3 | N/A | Regression with Large Language Models for Materials and Molecular Property Prediction  | Fine-tuning | Regression | 2024 | [DOI](https://doi.org/10.48550/arXiv.2409.06080) | [Code and data: part 1](https://figshare.com/articles/dataset/Regression_with_Large_Language_Models_for_Materials_and_Molecular_Property_Prediction_part_1_of_2_/26928439/1) [Code and data: part 2](https://figshare.com/articles/dataset/Regression_with_Large_Language_Models_for_Materials_and_Molecular_Property_Prediction_part_2_of_2_/26936770/1) |
| GPT 4 for Chemistry | GPT 4 | N/A | ExploringGPT-4’s potentialin chemical tasks, such as foundational chemistry knowledge, cheminformatics, data analysis,problem prediction, and proposal abilities | Prompt engineering | N/A | 2023 | [DOI](https://www.tandfonline.com/doi/epdf/10.1080/27660400.2023.2260300?needAccess=true)  | No github |
| N/A | GPT 3.5 | No clear data source | Polymer Solubility Prediction Using Large Language Models | Fine-tuning | Classification | 2025 | [DOI](https://doi.org/10.1021/acsmaterialslett.5c00054) | No github |
| N/A | Gemini 1.5 | [Github](https://github.com/xiaoyu961031/Fine-tuned-Gemini) | Can large language models predict the hydrophobicity of metal–organic frameworks? | Fine-tuning | Classification | 2025 | [DOI](https://doi.org/10.1039/D5TA01139F) | [Github](https://github.com/xiaoyu961031/Fine-tuned-Gemini) |
| LLM4Mat-bench | LLM-Prop, MatBERT, Llama 2-7b-chat,  | [Data and checkpoints](https://drive.google.com/drive/folders/1HpGhuNHG4EQCQMZaKPwEQNH9stJKw-ht?dmr%20=%201%26ec%20=%20wgc-drive-hero-goto) | Benchmarking large language models for materials property prediction | Fine-tuning, In-context learning | Classification Regression | 2025 | [DOI](https://doi.org/10.1088/2632-2153/add3bb) | [Github](https://github.com/vertaix/LLM4Mat-Bench) |
| PolyLLMem | Llama-3 Uni-Mol | [Github](https://github.com/zhangtr10/PolyLLMem) | Multimodal Machine Learning with Large Language Embedding Model for Polymer Property Prediction | Fine-tuning, LoRA | Regression | 2025 | [DOI](https://doi.org/10.1021/acs.chemmater.5c00940) | [Github](https://github.com/zhangtr10/PolyLLMem) |
| gptchem | GPT3 | The repository includes example scripts and [demo data sources](https://gptchem.readthedocs.io/en/latest/api.html#module-gptchem.data) | Natural-language pretrained | Prompt engineering, not for LoRA | Classification Regression | 2022 | [DOI](https://doi.org/10.1038/s42256-023-00788-1) | [Github](https://github.com/kjappelbaum/gptchem) |
| ChemLLM  | InternLM-2 | ChemData,ChemBench |  Pretrained for chemistry  | Fine-tuning,open weights(LoRA applicable)  | Classification | 2024 | [DOI](https://doi.org/10.48550/arXiv.2402.06852) | [Github](https://github.com/keyhsw/chemllm) |
| chemlift | General LLMs “The framework can be used with any HuggingFace model that accepts text input.” | N/A | Natural-language pretrained | Prompt engineering(Prompt \+ RAG) | Classification Regression | 2023 | [DOI](https://doi.org/10.1038/s42256-023-00788-1) | [Github](https://github.com/lamalab-org/chemlift) |
| De novo Design of Polymer Electrolytes with High Conductivity using GPT-based and Diffusion-based Generative Models | [minGPT](https://github.com/karpathy/minGPT)  | HTP-MD polymer-electrolyte dataset (SMILES) | Chemistry-pretrained SMILES-based generative | Fine-tuning hyperparameter grid search provided. (LoRA not described.) | Inverse design  | 2024 | [DOI](https://doi.org/10.1038/s41524-024-01470-9) | [Github](https://github.com/TRI-AMDD/PolyGen) |
| ChemLoRA | General LLMs”GPT-3 via API; any HuggingFace base such as GPT-2) with LoRA/PEFT adapters. | QM9-G4MP2 small-molecule dataset(SMILES)  | Natural-language pretrained | Prompt engineering (GPTChem-style prompts)  **,**Fine-tuning(LoRA) | Regression(molecular energy predictions) | 2023 | N/A | [Github](https://github.com/ankur56/ChemLoRA) |
| Text2Concrete | GPT-3.5-turbo | [Data](https://github.com/ghezalahmad/LLMs-for-the-Design-of-Sustainable-Concretes/tree/main/data) Known,Relevant,Random ( PubChem molecules) | Natural-language pretrained(Gaussian Process Regression and Random Forest for benchmarks) | Prompt engineering  | Regression | 2023 | [DOI](https://zenodo.org/records/8091195) | [Github](https://github.com/ghezalahmad/LLMs-for-the-Design-of-Sustainable-Concretes) |
| Molecule-Discovery-by-Context | ScholarBERT | literature corpus(for context) and Known,Relevant,Random ( PubChem molecules) | Natural-language pretrained | Prompt engineering(prompt, RAG) | Ranking for candidate discovery | 2023 | [DOI](https://doi.org/10.5281/zenodo.8122087) | [Github](https://github.com/tuhz/Molecule-Discovery-by-Context) |
| Using GPT-4 in parameter selection of polymer informatics: improving predictive accuracy amidst data scarcity and ‘Ugly Duckling’ dilemma  | GPT-4 | [Data](https://polymerdatabase.com/) | GPT4 is used to predict the refractive index of a molecule and select necessary descriptors.  Comparative study | N/A | Prediction Descriptor  | 2023 | [DOI](https://pubs.rsc.org/en/content/articlelanding/2023/DD/D3DD00138E) | [Github](https://github.com/KanHatakeyama/RefractiveIndexGPT) |
| SMI-TED-IC | SMI-TED | [Github](https://github.com/murtazazohair/IBM_SMI-TED-IC) | Chemical foundation model-guided design of high ionic conductivity electrolyte formulations | Fine-tuning | Regression | 2025 | [DOI](https://doi.org/10.1038/s41524-025-01774-4) | [Github](https://github.com/murtazazohair/IBM_SMI-TED-IC) |
| Leveraging GPT-4 to transform chemistry from paper to practice  | GPT-4 | [Data](https://gitlab.com/heingroup/gpt-xml-translation.) | 1\. Using GPT 4 to read scientific literature and generate actionable steps 2.Generate a script to communicate the steps to EasyMax Reactor | N/A | N/A | 2024 | N/A | [Github](https://gitlab.com/heingroup/gpt-xml-translation.) |
|ChemLLMBench|GPT-4, GPT-3, Davinci-003, Llama, Galactica | [Data](https://github.com/ChemFoundationModels/ChemLLMBench) | What can Large Language Models do in chemistry? A comprehensive benchmark on eight tasks |In-context learning | Generation, Classification, Ranking | 2023 | [DOI](https://doi.org/10.48550/arXiv.2305.18365) | [Github](https://github.com/ChemFoundationModels/ChemLLMBench)|


---

## Methodologies

Within this context, our project addresses three main tasks:

1. **Molecular property prediction**  
   Use LLMs to predict molecular properties, for example the conductivity of candidate electrolyte molecules.

2. **Classification**  
   Formulate classification problems such as distinguishing high-conductivity from low-conductivity molecules or grouping molecules by other experimentally relevant labels.

3. **Molecule generation**  
   Explore generative modelling, where the goal is to generate new SMILES strings that are chemically valid and have desirable predicted properties. These tasks together span both predictive and generative aspects of molecular design.

---

## Benchmark Results

Each group member re-implemented selected models from the literature, covering property prediction, classification, and molecule generation tasks. These implementations serve both as baselines and as starting points for small methodological variations and improvements.

We then experimented with commercial LLMs (e.g., GPT-4) and selected a common open-source backbone, **Llama-3.2-3B-Instruct**, as our main model. Across the three subtasks, each group member implements and evaluates the model against their respective literature baseline, and explores modifications to improve performance.

Code for the three subtasks is organised as:

- `./regression/` – LLM and baseline models for regression-style property prediction  
- `./classification/` – LLM and baseline models for molecular classification  
- `./inverse design/` – LLM-based SMILES generation and post-processing

---

## Future Work

- Investigating improved fine-tuning and evaluation strategies, including robustness in low-data regimes and better uncertainty estimates.
- Developing more targeted generative workflows that couple LLM-based generation with domain-specific filters or chemistry-based checks.

Contributions and suggestions are welcome, especially regarding datasets, baselines, and evaluation protocols.
