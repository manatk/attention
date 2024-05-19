import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

# Function to load masked forward results
def load_masked_forward_results(file_path, device):
    print(f"Loading masked forward results from {file_path}...")
    results = torch.load(file_path, map_location=device)
    print("Masked forward results loaded.")
    return results

'''
Function to evaluate the model on a given dataset

TODO: Currently score is just a random value - change so that the model is actually tested
'''
def evaluate_model(model, tokenizer, dataset, task, attention_masks, device):
    # Implement your evaluation logic here
    # This is a placeholder function to simulate benchmark scores
    print(f"Evaluating model on task: {task} with dataset: {dataset}...")

    if task == "BookSum":
        score = np.random.uniform(10, 30)  # Simulating perplexity score
    elif task == "LegalBENCH":
        score = np.random.uniform(0.5, 1.0)  # Simulating accuracy score
    elif task == "QuALITY":
        score = np.random.uniform(0.2, 0.8)  # Simulating accuracy score
    elif task == "The Stack":
        score = np.random.uniform(10, 30)  # Simulating perplexity score
    else:
        raise ValueError(f"Unknown task: {task}")

    print(f"Task: {task}, Dataset: {dataset}, Score: {score}")
    return score

'''
Function to plot benchmark scores for different alpha values

Params:
alpha_values - list of alpha values
benchmark_scores - scores associated with each alpha value
'''
def plot_alpha_summation_benchmarks(alpha_values, benchmark_scores, save_path):
    plt.figure()
    for dataset, scores in benchmark_scores.items():
        print(f"Plotting scores for {dataset}: {scores}")
        plt.plot(alpha_values, scores, label=dataset)
    plt.xlabel('Alpha Summation Values')
    plt.ylabel('Benchmark Scores')
    plt.title(f'Benchmark Scores vs. Alpha Summation Values')
    plt.legend()
    plt.savefig(save_path)
    plt.show()

'''
Function to plot benchmark scores for different token retention counts

Params:
alpha_values - list of alpha values
benchmark_scores - scores associated with each alpha value
'''
def plot_token_retention_benchmarks(token_counts, benchmark_scores, save_path):
    plt.figure()
    for dataset, scores in benchmark_scores.items():
        print(f"Plotting scores for {dataset}: {scores}")
        plt.plot(token_counts, scores, label=dataset)
    plt.xlabel('Number of Tokens Retained')
    plt.ylabel('Benchmark Scores')
    plt.title(f'Benchmark Scores vs. Token Retention Counts')
    plt.legend()
    plt.savefig(save_path)
    plt.show()

# Main function to execute the evaluation and plotting
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model_path = "gpt2"
    access_token = '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'
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_path, token=access_token).to(device)
    print("Tokenizer and model loaded.")

    results_path = './Baseline_Tensors/gpt2_masked_forward_results.pth'
    masked_forward_results = load_masked_forward_results(results_path, device)

    alpha_values = [0.5, 0.6, 1.0]
    token_counts = [10, 20, 30]  # Example token counts for retention

    tasks = ["BookSum", "LegalBENCH", "QuALITY", "The Stack"]
    datasets = ["Dataset1", "Dataset2", "Dataset3"]  # Placeholder for actual dataset names

    # Evaluate benchmarks for different alpha summation values
    print("Evaluating benchmarks for different alpha summation values...")
    alpha_benchmark_scores = {dataset: [0] * len(alpha_values) for dataset in datasets}
    for alpha_idx, alpha in enumerate(alpha_values):
        print(f"Evaluating for alpha value: {alpha}")
        for dataset in datasets:
            print(f"  Evaluating dataset: {dataset}")
            for task in tasks:
                score = evaluate_model(model, tokenizer, dataset, task, masked_forward_results, device)
                alpha_benchmark_scores[dataset][alpha_idx] += score / len(tasks)
    print("Evaluation for alpha summation values completed.")
    plot_alpha_summation_benchmarks(alpha_values, alpha_benchmark_scores, "alpha_summation_benchmarks.png")

    # Evaluate benchmarks for different token retention counts
    print("Evaluating benchmarks for different token retention counts...")
    token_benchmark_scores = {dataset: [0] * len(token_counts) for dataset in datasets}
    for count_idx, count in enumerate(token_counts):
        print(f"Evaluating for token retention count: {count}")
        for dataset in datasets:
            print(f"  Evaluating dataset: {dataset}")
            for task in tasks:
                score = evaluate_model(model, tokenizer, dataset, task, masked_forward_results, device)
                token_benchmark_scores[dataset][count_idx] += score / len(tasks)
    print("Evaluation for token retention counts completed.")
    plot_token_retention_benchmarks(token_counts, token_benchmark_scores,  "token_retention_benchmarks.png")

if __name__ == '__main__':
    print("Starting evaluation and plotting...")
    main()
    print("Evaluation and plotting completed.")
