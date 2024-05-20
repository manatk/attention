import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score

# Function to load masked forward results
def load_masked_forward_results(file_path):
    print(f"Loading masked forward results from {file_path}...")
    results = torch.load(file_path)
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
def evaluate_accuracy(model, tokenizer, dataset, device):
    model.eval()
    predictions, references = [], []
    with torch.no_grad():
        for example in dataset:
            inputs = tokenizer(example['question'], example['context'], return_tensors='pt', truncation=True).to(device)
            outputs = model(**inputs)
            predictions.append(torch.argmax(outputs.logits, dim=-1).item())
            references.append(example['answers']['text'][0])
    return accuracy_score(references, predictions)

# Function to evaluate BookSum dataset
def evaluate_booksum(model, tokenizer, dataset, device):
    texts = [example['article'] for example in dataset]
    return calculate_perplexity(model, tokenizer, texts, device)

# Function to evaluate LegalBENCH dataset
def evaluate_legalbench(model, tokenizer, dataset, device):
    return evaluate_accuracy(model, tokenizer, dataset, device)

# Function to evaluate QuALITY dataset
def evaluate_quality(model, tokenizer, dataset, device):
    return evaluate_accuracy(model, tokenizer, dataset, device)

# Function to evaluate The Stack dataset
def evaluate_the_stack(model, tokenizer, dataset, device):
    texts = [example['func'] for example in dataset]
    return calculate_perplexity(model, tokenizer, texts, device)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device set to {device}")

    model_path = "gpt2"
    access_token = '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'

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_path, use_auth_token=access_token).to(device)
    print("Tokenizer and model loaded.")

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

    # Mapping from tasks to their respective evaluation functions
    evaluation_functions = {
        "BookSum": evaluate_booksum,
        "LegalBENCH": evaluate_legalbench,
        "QuALITY": evaluate_quality,
        "The Stack": evaluate_the_stack
    }

    # Evaluate benchmarks for different alpha summation values
    print("Evaluating benchmarks for different alpha summation values...")
    alpha_benchmark_scores = {dataset: [0] * len(alpha_values) for dataset in datasets.keys()}
    for dataset_name, dataset in datasets.items():
        for alpha_idx, alpha in enumerate(alpha_values):
            print(f"Evaluating for alpha value: {alpha} on dataset: {dataset_name}")
            total_score = sum(evaluation_functions[task](model, tokenizer, dataset, device) for task in tasks)
            alpha_benchmark_scores[dataset_name][alpha_idx] = total_score / len(tasks)
    print("Evaluation for alpha summation values completed.")
    plot_alpha_summation_benchmarks(alpha_values, alpha_benchmark_scores, "Alpha Summation Benchmarks", "alpha_summation_benchmarks_gpt2.png")

    # Evaluate benchmarks for different token retention counts
    print("Evaluating benchmarks for different token retention counts...")
    token_benchmark_scores = {dataset: [0] * len(token_counts) for dataset in datasets.keys()}
    for dataset_name, dataset in datasets.items():
        for count_idx, count in enumerate(token_counts):
            print(f"Evaluating for token retention count: {count} on dataset: {dataset_name}")
            total_score = sum(evaluation_functions[task](model, tokenizer, dataset, device) for task in tasks)
            token_benchmark_scores[dataset_name][count_idx] = total_score / len(tasks)
    print("Evaluation for token retention counts completed.")
    plot_token_retention_benchmarks(token_counts, token_benchmark_scores, "Token Retention Benchmarks", "token_retention_benchmarks_gpt2.png")

if __name__ == '__main__':
    print("Starting evaluation and plotting...")
    main()
    print("Evaluation and plotting completed.")
