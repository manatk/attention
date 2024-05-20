import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score
import multiprocessing as mp
import json
import os

# Function to load masked forward results
def load_masked_forward_results(file_path):
    print(f"Loading masked forward results from {file_path}...")
    results = torch.load(file_path, map_location='cpu')
    print("Masked forward results loaded.")
    return results

# Function to evaluate perplexity
def calculate_perplexity(model, tokenizer, texts, device):
    model.eval()
    total_loss = 0.0
    total_length = 0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=model.config.n_positions).to(device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
            total_length += inputs["input_ids"].size(1)
    return np.exp(total_loss / total_length)

# Function to evaluate accuracy
def evaluate_accuracy(model, tokenizer, dataset, device, question_key, context_key, answer_key):
    model.eval()
    predictions, references = [], []
    with torch.no_grad():
        for example in dataset:
            inputs = tokenizer(example[question_key], example[context_key], return_tensors='pt', truncation=True).to(device)
            outputs = model(**inputs)
            predictions.append(torch.argmax(outputs.logits, dim=-1).item())
            references.append(example[answer_key][0]['text'] if isinstance(example[answer_key], dict) else example[answer_key])
    return accuracy_score(references, predictions)

# Function to evaluate the model on a given dataset and task
def evaluate_model(args):
    model_path, access_token, dataset, task, attention_masks, device, dataset_name, question_key, context_key, answer_key = args

    print(f"Loading tokenizer and model for {dataset_name} on task: {task}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_path, use_auth_token=access_token).to(device)
    print(f"Tokenizer and model loaded for {dataset_name} on task: {task}.")

    print(f"Evaluating model on task: {task} with dataset: {dataset_name}...")

    if task == "BookSum":
        texts = [example['article'] for example in dataset]
        score = calculate_perplexity(model, tokenizer, texts, device)
    elif task == "LegalBENCH":
        score = evaluate_accuracy(model, tokenizer, dataset, device, question_key, context_key, answer_key)
    elif task == "QuALITY":
        score = evaluate_accuracy(model, tokenizer, dataset, device, question_key, context_key, answer_key)
    elif task == "The Stack":
        texts = [example['func'] for example in dataset]
        score = calculate_perplexity(model, tokenizer, texts, device)
    else:
        raise ValueError(f"Unknown task: {task}")

    print(f"Task: {task}, Dataset: {dataset_name}, Score: {score}")
    return (dataset_name, task, score)

# Function to plot benchmark scores for different alpha values
def plot_alpha_summation_benchmarks(alpha_values, benchmark_scores, task, save_path):
    print(f"Plotting alpha summation benchmarks for task: {task}...")
    plt.figure()
    for dataset, scores in benchmark_scores.items():
        print(f"  Plotting scores for {dataset}: {scores}")
        plt.plot(alpha_values, scores, label=dataset)
    plt.xlabel('Alpha Summation Values')
    plt.ylabel('Benchmark Scores')
    plt.title(f'Benchmark Scores vs. Alpha Summation Values for {task}')
    plt.legend()
    plt.savefig(save_path)
    plt.show()
    print(f"Alpha summation benchmarks plot saved to {save_path}.")

# Function to plot benchmark scores for different token retention counts
def plot_token_retention_benchmarks(token_counts, benchmark_scores, task, save_path):
    print(f"Plotting token retention benchmarks for task: {task}...")
    plt.figure()
    for dataset, scores in benchmark_scores.items():
        print(f"  Plotting scores for {dataset}: {scores}")
        plt.plot(token_counts, scores, label=dataset)
    plt.xlabel('Number of Tokens Retained')
    plt.ylabel('Benchmark Scores')
    plt.title(f'Benchmark Scores vs. Token Retention Counts for {task}')
    plt.legend()
    plt.savefig(save_path)
    plt.show()
    print(f"Token retention benchmarks plot saved to {save_path}.")

# Main function to execute the evaluation and plotting
def main():
    print("Setting device...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device set to {device}")

    model_path = "gpt2"
    access_token = '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'

    results_path = './Baseline_Tensors/gpt2_masked_forward_results.pth'
    masked_forward_results = load_masked_forward_results(results_path)

    alpha_values = [0.5, 0.6, 1.0]
    token_counts = [10, 20, 30]  # Example token counts for retention

    tasks = ["BookSum", "LegalBENCH", "QuALITY", "The Stack"]
    print("Loading datasets...")
    datasets = {
        "BookSum": load_dataset('cnn_dailymail', '3.0.0', split='test'),
        "LegalBENCH": load_dataset('squad', split='validation'),
        "QuALITY": load_dataset('race', 'all', split='test'),
        "The Stack": load_dataset('code_x_glue_cc_clone_detection_big_clone_bench', split='validation')
    }
    print("Datasets loaded.")

    # Prepare arguments for multiprocessing
    eval_args = []
    for dataset_name in datasets.keys():
        for task in tasks:
            if dataset_name == "LegalBENCH" or dataset_name == "QuALITY":
                question_key = "question" if "question" in datasets[dataset_name].column_names else "title"
                context_key = "context" if "context" in datasets[dataset_name].column_names else "content"
                answer_key = "answers" if "answers" in datasets[dataset_name].column_names else "label"
            else:
                question_key = context_key = answer_key = None  # Not used for other datasets
            eval_args.append((model_path, access_token, datasets[dataset_name], task, masked_forward_results, device, dataset_name, question_key, context_key, answer_key))

    # Function to save evaluation results
    def save_evaluation_results(results, filename):
        with open(filename, 'w') as f:
            json.dump(results, f)
        print(f"Saved evaluation results to {filename}.")

    # Parallelize evaluation for alpha summation values
    print("Evaluating benchmarks for different alpha summation values...")
    alpha_benchmark_scores = {dataset: [0] * len(alpha_values) for dataset in datasets.keys()}
    with mp.Pool(processes=min(2, mp.cpu_count())) as pool:  # Cap the number of processes at 2
        for alpha_idx, alpha in enumerate(alpha_values):
            print(f"Evaluating for alpha value: {alpha}")
            results = pool.map(evaluate_model, eval_args)
            for dataset_name, task, score in results:
                alpha_benchmark_scores[dataset_name][alpha_idx] += score / len(tasks)
    print("Evaluation for alpha summation values completed.")
    save_evaluation_results(alpha_benchmark_scores, "alpha_benchmark_scores.json")
    plot_alpha_summation_benchmarks(alpha_values, alpha_benchmark_scores, "Alpha Summation Benchmarks", "alpha_summation_benchmarks_gpt2.png")

    # Parallelize evaluation for token retention counts
    print("Evaluating benchmarks for different token retention counts...")
    token_benchmark_scores = {dataset: [0] * len(token_counts) for dataset in datasets.keys()}
    with mp.Pool(processes=min(2, mp.cpu_count())) as pool:  # Cap the number of processes at 2
        for count_idx, count in enumerate(token_counts):
            print(f"Evaluating for token retention count: {count}")
            results = pool.map(evaluate_model, eval_args)
            for dataset_name, task, score in results:
                token_benchmark_scores[dataset_name][count_idx] += score / len(tasks)
    print("Evaluation for token retention counts completed.")
    save_evaluation_results(token_benchmark_scores, "token_benchmark_scores.json")
    plot_token_retention_benchmarks(token_counts, token_benchmark_scores, "Token Retention Benchmarks", "token_retention_benchmarks_gpt2.png")

if __name__ == '__main__':
    print("Starting evaluation and plotting...")
    main()
    print("Evaluation and plotting completed.")
