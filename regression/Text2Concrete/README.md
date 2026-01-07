# From Text to Concrete: Developing Sustainable Concretes with In-Context Learning

## Introduction
This repository contains the code and dataset for the study "From Text to Concrete: Developing Sustainable Concretes with In-Context Learning". The project aims to improve the development of sustainable concrete formulations using in-context learning (ICL) and large language models (LLMs). By leveraging the potential of LLMs, this research aims to overcome the limitations of traditional methods and accelerate the discovery of novel, sustainable, and high-performance materials.

### Credit
This repository is originally a fork, based on previous work that can be found at https://github.com/ghezalahmad/LLMs-for-the-Design-of-Sustainable-Concretes.

### Changes
As the text-davinci-003 model is deprecated, it is no longer used in this project. Instead, we test several LLMs on the original data, both closed- and open-source models. The LLMs that we use for benchmarking are:

- GPT-3.5-turbo-instruct
- GPT-4o-mini
- Llama-3.2-3B-Instruct
- Llama-3.1-70B-Instruct-bnb-4bit

We have made some changes to the code, and the software and packages that were used. Finally, we slightly altered the given context prompts, and calculations of the metrics used to evaluate the model performance.

### Overview

The primary goal of this study is to compare the prediction performance of compressive strength using ICL and LLMs against established methods such as Gaussian Process Regression (GPR) and Random Forest (RF). The dataset comprises 240 alternative and more sustainable concrete formulations based on fly ash and ground granulated slag binders, along with their respective compressive strengths.

## Table of Contents

- [Installation](#installation)
- [Post-installation](#post-installation)
- [Hardware](#hardware)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation

1. Clone the repository to your local machine using `git clone`:

    ```bash
    git clone https://github.com/username/repo.git
    ```

2. Change directory to the cloned repository:

    ```bash
    cd repo
    ```


We provide two different options for the rest of the setup

### Option 1: Using Apptainer to build and run a container on a Linux filesystem (recommended)

3. Change directory to container-recipes

    ```bash
    cd container-recipes
    ```

4. Build the container with Apptainer using the definition file
    ```bash
    apptainer build my-container.sif container-recipe.def
    ```

### Option 2: Download dependencies on a python environment using pip (not tested)

3. Install the dependencies using a package manager such as `npm` or `pip`:
    ```bash
    pip install -r container-recipes/requirements.txt
    ```


## Post-installation

5. Start a jupyter notebook

    Using Apptainer:
    ```bash
    apptainer exec --nv path-to-container jupyter notebook
    ```
    Using a python environment:
     ```bash
    jupyter notebook
    ```

6. Open a web browser and navigate to `http://localhost:3000` to access the application.
7. Create an OpenAI API key from the [OpenAI](https://platform.openai.com/api-keys) website, and follow the [safety instructions](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety).
9. Prepare a CSV file with prompts and completions as input data. Update the file path in the code to point to your CSV file.
10. For the Llama models, update the model file path to the local file path, or alternatively the huggingface model ID to download it.

## Hardware
For inference of Llama-3.2-3B-Instruct and Llama-3.1-70B-Instruct-bnb-4bit we used a T4 and A40 GPU respectively.

## Usage
1. Run the different benchmarking notebooks in your container or environment to train the completion predictor and generate completions for test data.
2. The completion predictor's performance will be evaluated using R-squared, MSE, and MAE metrics, which will be printed on the console. The resulting predictions will be saved as CSVs in the results folder for each model. 
3. Run the benchmarking results notebooks for the evaluation of each model.
3. You can modify the code to customize the prompts, completions, and other parameters as needed for your specific use case.


## Contributing
Contributions to this project are welcome! If you would like to contribute, please follow standard GitHub practices, such as forking the repository, creating a branch for your changes, and submitting a pull request with a clear description of your changes. Please ensure that your changes are well-tested and adhere to the project's coding standards.

## License
This software is released under the [MIT License](https://opensource.org/licenses/MIT).

## Acknowledgements

The following resources were used in the development of this project:

- [OpenAI](https://openai.com) for providing the GPT API that powers the completion predictor.
- [Pandas](https://pandas.pydata.org/) and [scikit-learn](https://scikit-learn.org/) libraries for data manipulation and machine learning capabilities in Python.

