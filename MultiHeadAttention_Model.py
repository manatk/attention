from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Model, GPT2Config
import torch
import numpy as np
import argparse
import pdb
import trainer
import models  # Ensure this points to the correct file containing your custom models

print("Starting script...")

model_path = "gpt2"
access_token = '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'
print("Model path and access token set.")

def create_attention_mask(attentions, threshold):
    print("Creating attention masks...")
    batch_size, num_heads, seq_length = attentions[0].size(0), attentions[0].size(1), attentions[0].size(2)
    attentions_np = np.stack([attn.detach().cpu().numpy() for attn in attentions])
    sums = np.sum(attentions_np, axis=3)
    masks = np.where(sums < threshold * seq_length, 0, 1)
    attention_masks = np.repeat(masks[:, :, :, np.newaxis], seq_length, axis=3)
    print("Attention masks created.")
    return torch.from_numpy(attention_masks).to(attentions[0].device)

print("Setting up argument parser...")
parser = argparse.ArgumentParser(description='Choose whether to finetune or evaluate.')
parser.add_argument('threshold', metavar='threshold', type=float, nargs='?', default=0.5,
                    help='Threshold t such that, if sum(alpha_w) > t, for given w, model is said to attend to w')
parser.add_argument('finetune', metavar='finetune', type=bool, nargs='?', default=False, help='Whether to finetune model or not')
print("Argument parser set up.")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("Tokenizer loaded.")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_path, output_attentions=True, use_auth_token=access_token)
print("Model loaded.")

args = parser.parse_args()
print(f"Arguments parsed: {args}")

def main():
    print("Entering main function...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    prompt = "What does this code do?"
    print(f"Tokenizing prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=model.config.n_positions).to(device)
    print(f"Inputs: {inputs}")
    print("Moving model to device...")
    model.to(device)
    print("Generating model outputs...")
    outputs = model(**inputs)
    attentions = outputs.attentions
    print(f"Model outputs generated. Attentions: {attentions}")
    print("Creating attention masks...")
    attention_masks = create_attention_mask(attentions, args.threshold).to(device)
    #print(f"Attention masks: {attention_masks}")

    input_ids = inputs['input_ids']
    #print(f"Generating embeddings for input IDs: {input_ids}")
    embeddings = model.get_input_embeddings()(input_ids)
    embeddings = embeddings.to(device)
    #print(f"Embeddings: {embeddings}")

    print("Loading custom model configuration...")
    config = models.GPTConfig(50257, 1024, n_layer=len(attention_masks), n_head=attention_masks.size()[2], n_embd=embeddings.size()[2])
    config.attention_masks = attention_masks
    model_custom = models.CustomGPT(config)
    model_custom = model_custom.to(device)
    print("Custom model loaded and moved to device.")
    print("Running custom forward pass...")
    output = model_custom(embeddings)[0]
    print("Custom forward pass completed.")

    print("Final Model Output:")
    token_ids = torch.argmax(output, dim=-1)

    print(output)
    decoded_output = tokenizer.decode(token_ids[0], skip_special_tokens=True)

    print("Final Model Output:")
    print(decoded_output)

    print("Main function completed.")

if __name__ == '__main__':
    print("Executing script...")
    main()
    print("Script execution completed.")
