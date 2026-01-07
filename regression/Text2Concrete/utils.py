import pandas as pd
import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from IPython.display import Markdown, display
from openai import OpenAI
from openai import RateLimitError
import random
import re
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def read_and_evaluate(folder_name, model_name):
    metrics = {"R-squared" : [], "MSE": [], "MAE": []}
    for i in range(10):
        file_path = os.path.join("results", model_name, folder_name, "test.csv")
        df = pd.read_csv(file_path)
        if folder_name in ["GPR", "RF"]:
            df_iter = df[df["Iteration"] == i+1]
            true_values = df_iter["True Values"]
            predicted_values = df_iter["Predicted Values"]
        else:
            df_iter = df[df["iteration"] == i+1]
            true_values = df_iter["True Value"]
            predicted_values = df_iter["Prediction"]

        r_squared = r2_score(true_values, predicted_values)
        mse = mean_squared_error(true_values, predicted_values)
        mae = mean_absolute_error(true_values, predicted_values)
        metrics["R-squared"].append(r_squared)
        metrics["MSE"].append(mse)
        metrics["MAE"].append(mae)
    return metrics

def gather_metric_results(approaches, model_name):
    results = {}
    for approach in approaches:
        metrics = read_and_evaluate(approach, model_name)
        approach_results = {}
        for metric, values in metrics.items():
            metric_results = {}
            mean, margin_error = compute_mean_and_margin_error(values)
            metric_results["mean"] = mean
            metric_results["margin_error"] = margin_error
            approach_results[metric] = metric_results
        results[approach] = approach_results
    return results

def compute_mean_and_margin_error(metric_sample):
    n = len(metric_sample)
    mean = np.mean(metric_sample)
    std = np.std(metric_sample)
    standard_error = std/np.sqrt(n)
    
    confidence_level = 0.95
    degrees_of_freedom = n - 1

    lower_ci, upper_ci = stats.t.interval(confidence_level,
                                          degrees_of_freedom,
                                          mean,
                                          standard_error)
    margin_error = upper_ci - mean
    return mean, margin_error


def plot_scatter(model_name, folder_name, ax):
    file_path = os.path.join("results", model_name, folder_name, "test.csv")
    df = pd.read_csv(file_path)
    if folder_name in ["GPR", "RF"]:
        true_values = df["True Values"]
        predicted_values = df["Predicted Values"]
    else:
        true_values = df["True Value"]
        predicted_values = df["Prediction"]

    ax.scatter(true_values, predicted_values, label=folder_name)
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.legend()
    ax.set_title(folder_name)

def create_markdown_table(results):
    markdown_table = "| Approach | R-squared | MSE | MAE |\n"
    markdown_table += "| -------- | --------- | --- | --- |\n"

    for approach, result in results.items() :
        r_squared_result = result["R-squared"]
        mse_result = result["MSE"]
        mae_result = result["MAE"]

        r_squared_mean = r_squared_result["mean"]
        r_squared_margin = r_squared_result["margin_error"]

        mse_mean = mse_result["mean"]
        mse_margin = mse_result["margin_error"]
        
        mae_mean = mae_result["mean"]
        mae_margin = mae_result["margin_error"]

        markdown_table += f"| {approach} | {r_squared_mean:.2f} +- {r_squared_margin:.2f} | {mse_mean:.2f} +- {mse_margin:.2f} | {mae_mean:.2f} +- {mae_margin:.2f} |\n"

    display(Markdown("## Benchmarking Results"))
    display(Markdown(markdown_table))

def plot_actual_vs_predicted(model_name):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    plot_scatter(model_name, "RF", axes[0])
    plot_scatter(model_name, "GPR", axes[1])
    plot_scatter(model_name, "ICL_finetuned", axes[2])
    plot_scatter(model_name, "ICL", axes[3])

    plt.tight_layout()
    plt.show()



def generate_prediction(client, train, test, model_name, model = None, tokenizer = None):
        # Loop in case of RateLimitError
        for i in range(5):
            response = ""
            result_text = ""
            try:
                if model_name == "gpt-3.5-turbo-instruct":
                    prompt = train + test + " ; completion: "
                    response = generate_completion(client, model_name, prompt)
                    result_text = response.choices[0].text.strip()
                else:
                    train_messages = train
                    test_message = [{"role":"user", "content": test}]
                    messages = train_messages + test_message
                    
                    if model_name == "gpt-4o-mini":
                        response = generate_chat_completion(client, model_name, messages)
                        result_text = response.choices[0].message.content
                    else:
                        result_text = generate_llama_output(model, tokenizer, messages)
                
                print(f"response: {result_text}")
                result_text = re.sub("[^0-9.]", "", result_text)
                print(f"result: {float(result_text)}")
                return float(result_text)
            
            except RateLimitError as e:
                print(e.message)
                print(f"Rate limit error: Waiting for 60s")
                time.sleep(60)

def generate_completion(client, model_name, prompt):
    response = client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    max_tokens=10,
                    n=1,
                    stop=None,
                    temperature=0,
                )
    return response

def generate_chat_completion(client, model_name, messages):
    response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=10,
                n=1,
                stop=None,
                temperature=0,
            )
    return response

def generate_llama_output(model, tokenizer, messages):
    
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt = True, return_tensors="pt", return_dict = True
    ).to(model.device)

    outputs = model.generate(**inputs, 
                             max_new_tokens = 10,
                             temperature = 0,
                             num_return_sequences = 1,
                             do_sample = False
    )
    
    prompt_length = inputs["input_ids"].shape[-1]
    response_ids = outputs[0][prompt_length:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)

    return response


def save_results_to_csv(model_name, approach, result_list):
    
    # Relative path where the results from current approach will be saved  
    result_path= os.path.join('results', model_name, approach)

    # Create needed directories if they do not already exist
    #dir_name = os.path.dirname(result_path)
    os.makedirs(result_path, exist_ok=True)

    # Concatenate all the results to a single DataFrame
    result_df = pd.concat([pd.DataFrame(result_dict) for result_dict in result_list], ignore_index=True)

    # Save the results to a single CSV file
    result_path = os.path.join('results', model_name, approach, 'test.csv')
    result_df.to_csv(result_path, index=False)

    test_sample_size = len(result_df[result_df["iteration"] == 1])
    
    print(f"Results for {test_sample_size*10} iterations are saved to a single CSV file.")

def sample_train_and_test_data(data, train_size, test_size, indices):
    random.shuffle(indices)
    #random_indices = np.random.choice(len(data), 
                                    #train_size+test_size, 
                                    #False)
    
    train_data = [data[idx] for idx in indices[:train_size]]
    test_data = [data[idx] for idx in indices[train_size:train_size + test_size]]
    return train_data, test_data

def create_training_messages(training_prompts, system_message):
    training_messages = []
    training_messages.append(system_message)

    for input_prompt, output_prompt in training_prompts:
        #Strip whitespace
        input_prompt = input_prompt.strip()
        output_prompt = output_prompt.strip()

        user_message = {"role": "user", "content": input_prompt}
        assistant_message = {"role": "assistant", "content" : output_prompt}
        training_messages.append(user_message)
        training_messages.append(assistant_message)
    
    return training_messages


def predict_and_evaluate(client, train_prompts, test_prompts, true_values, model_name, model = None, tokenizer = None):
        predictions = [generate_prediction(client, 
                                       train_prompts, 
                                       test_prompt, 
                                       model_name,
                                       model,
                                       tokenizer) for test_prompt in test_prompts]
        r2 = r2_score(true_values, predictions)
        mae = mean_absolute_error(true_values, predictions)
        mse = mean_squared_error(true_values, predictions)
        
        # Evaluation 
        print(f"R-squared: {r2:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")
        
        return predictions

def append_to_result_list(test_prompts, true_values, predictions, result_list):
        N = len(test_prompts)
        iteration = len(result_list) + 1
        result_dict = {'Test Prompt': test_prompts, 
                   'True Value': true_values, 
                   'Prediction': predictions, 
                   'iteration': [iteration]*N}
        result_list.append(result_dict)


def gather_LLM_results(data, train_size, test_size, client, model_name, indices, system_message = None, context_prompt = None, model=None, tokenizer = None):
    train_data, test_data = sample_train_and_test_data(data,
                                                       train_size,
                                                       test_size,
                                                       indices)
    
    test_prompts = [line.split("; completion:")[0] for line in test_data]
    true_values = [float(line.split("; completion:")[1]) for line in test_data]
    
    if model_name == "gpt-3.5-turbo-instruct":
        training_prompts = [line.strip() for line in train_data]
        training_text = "\n".join(training_prompts)
        train_input = f"{context_prompt}\n\n{training_text}\n"
    else:
        training_prompts = [line.split("; completion:") for line in train_data]
        train_input = create_training_messages(training_prompts, system_message)
    
    predictions = predict_and_evaluate(client, 
                                       train_input, 
                                       test_prompts,
                                       true_values,
                                       model_name,
                                       model,
                                       tokenizer)
    
    return test_prompts, true_values, predictions