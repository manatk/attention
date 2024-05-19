from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Model, GPT2Config
import torch
import numpy as np
import argparse

print("Starting script...")

model_path = "gpt2"
access_token = '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'
print("Model path and access token set.")

'''
Creates attention mask for all layers such that all words whose sum of alpha values is less than threshold are dropped out.

Params:
attentions (Tuple of tensors) - Attention matrix that model outputs
threshold - threshold under which elements are dropped out

Returns tensor of size [num_layers, batch_size, num_heads, seq_length, seq_length]
'''
def create_attention_mask(attentions, threshold):
    print("Creating attention masks...")
    batch_size, num_heads, seq_length = attentions[0].size(0), attentions[0].size(1), attentions[0].size(2)
    attentions_np = np.stack([attn.detach().cpu().numpy() for attn in attentions])
    sums = np.sum(attentions_np, axis=3)
    masks = np.where(sums < threshold, 0, 1)
    attention_masks = np.repeat(masks[:, :, :, np.newaxis], seq_length, axis=3)
    print("Attention masks created.")
    return torch.from_numpy(attention_masks).to(attentions[0].device)


class CustomGPT2Model(GPT2Model):
    def __init__(self, config):
        print("Initializing CustomGPT2Model...")
        super(CustomGPT2Model, self).__init__(config)
        print("CustomGPT2Model initialized.")
    
    def forward(self, attention_masks, inputs_embeds=None, **kwargs):
        print("Starting forward pass in CustomGPT2Model...")
        all_hidden_states = []
        hidden_states = inputs_embeds
        
        for i, layer_module in enumerate(self.h):
            print(f"Processing layer {i}...")
            if attention_masks is not None and i < len(attention_masks):
                attention_mask = attention_masks[i]
                print(f"Applied attention mask for layer {i}.")
            else:
                attention_mask = None
            
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                **kwargs
            )
            hidden_states = layer_outputs[0]
            all_hidden_states.append(hidden_states)
            print(f"Layer {i} processed.")
        
        hidden_states = self.ln_f(hidden_states)
        all_hidden_states.append(hidden_states)
        print("Forward pass in CustomGPT2Model completed.")
        return hidden_states, all_hidden_states

print("Setting up argument parser...")
parser = argparse.ArgumentParser(description='Choose whether to finetune or evaluate.')
parser.add_argument('threshold', metavar='threshold', type=float, nargs='?', default=0.5,
                    help='Threshold t such that, if sum(alpha_w) > t, for given w, model is said to attend to w')
parser.add_argument('finetune', metavar='finetune', type=bool, nargs='?', default=False, help='Whether to finetune model or not')
print("Argument parser set up.")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, force_download=True, token=access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("Tokenizer loaded.")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_path, output_attentions=True, token=access_token)
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
    attention_masks = create_attention_mask(attentions, args.threshold)
    print(f"Attention masks: {attention_masks}")

    input_ids = inputs['input_ids']
    print(f"Generating embeddings for input IDs: {input_ids}")
    embeddings = model.get_input_embeddings()(input_ids)
    print(f"Embeddings: {embeddings}")

    print("Loading custom model configuration...")
    config = GPT2Config.from_pretrained(model_path)
    model_custom = CustomGPT2Model.from_pretrained(model_path, config=config).to(device)
    print("Custom model loaded and moved to device.")

    print("Running custom forward pass...")
    with torch.no_grad():
        final_output, all_hidden_states = model_custom(
            inputs_embeds=embeddings,
            attention_masks=attention_masks
        )
    print("Custom forward pass completed.")

    # Store results/tensors of the second forward pass
    model_name = model_path.replace('/', '_')
    masked_forward_results = {f'layer_{i}': hidden_state for i, hidden_state in enumerate(all_hidden_states)}
    save_path = f'{model_name}_masked_forward_results.pth'
    print(f"Storing masked forward results to {save_path}...")
    torch.save(masked_forward_results, save_path)
    print(f"Masked forward results saved to {save_path}.")

    print("Final Model Output:")
    print(final_output)
    print("All Hidden States:")
    for i, hidden_state in enumerate(all_hidden_states):
        print(f"Layer {i} Hidden State Shape:", hidden_state.shape)
    print("Main function completed.")

if __name__ == '__main__':
    print("Executing script...")
    main()
    print("Script execution completed.")
