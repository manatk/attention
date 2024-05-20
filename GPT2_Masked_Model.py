from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
import numpy as np
import argparse

print("Starting script...")

model_path = "gpt2"
access_token = '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'
print("Model path and access token set.")

def create_attention_mask(attentions, threshold):
    print("Creating attention masks...")
    num_layers = len(attentions)
    attention_masks = []

    for layer_attn in attentions:
        # layer_attn is [batch_size, num_heads, seq_length, seq_length]
        batch_size, num_heads, seq_length, _ = layer_attn.shape
        layer_attn_np = layer_attn.detach().cpu().numpy()
        sums = np.sum(layer_attn_np, axis=-1)
        mask = np.where(sums < threshold, 0, 1)
        attention_mask = np.repeat(mask[:, :, :, np.newaxis], seq_length, axis=-1)
        attention_masks.append(torch.from_numpy(attention_mask).to(layer_attn.device))

    print("Attention masks created.")
    return attention_masks


class CustomGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        print("Initializing CustomGPT2LMHeadModel...")
        super(CustomGPT2LMHeadModel, self).__init__(config)
        print("CustomGPT2LMHeadModel initialized.")

    def forward(self, input_ids=None, past_key_values=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, attention_masks=None):
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=True,  # Output attentions
            output_hidden_states=output_hidden_states,
            return_dict=True,  # Ensure outputs are returned as a dictionary-like object
        )
        
        hidden_states = transformer_outputs.last_hidden_state
        all_attentions = transformer_outputs.attentions

        # Apply custom attention masks to each layer's attention weights
        if attention_masks is not None:
            for i in range(len(all_attentions)):
                all_attentions[i] = all_attentions[i] * attention_masks[i]
                print(f"Layer {i} attention mask applied")

        lm_logits = self.lm_head(hidden_states)
        
        return CausalLMOutputWithPast(
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=all_attentions,  # Use the modified attentions
        )

print("Setting up argument parser...")
parser = argparse.ArgumentParser(description='Choose whether to finetune or evaluate.')
parser.add_argument('threshold', metavar='threshold', type=float, nargs='?', default=0.5,
                    help='Threshold t such that, if sum(alpha_w) < t, tokens are masked out')
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

    prompt = "My name is Manat Kaur and I am 20 years old I am trying to work on this project but I am confused. Can you help me?"
    print(f"Tokenizing prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=model.config.n_positions).to(device)
    print(f"Inputs: {inputs}")

    print("Moving model to device...")
    model.to(device)
    print("Generating model outputs to obtain attentions...")
    outputs = model(**inputs)
    attentions = outputs.attentions
    print(f"Model outputs generated. Attentions: {attentions}")

    print("Creating custom attention masks...")
    custom_attention_masks = create_attention_mask(attentions, args.threshold)
    print(len(custom_attention_masks[0]))
    print(f"Custom attention masks created: {custom_attention_masks}")

    print("Loading custom model configuration...")
    config = GPT2Config.from_pretrained(model_path)
    model_custom = CustomGPT2LMHeadModel.from_pretrained(model_path, config=config).to(device)
    print("Custom model loaded and moved to device.")

    # Generate text using the custom attention masks
    print("Generating text with the custom model using custom attention masks...")
    print(custom_attention_masks)
    generated_ids = model_custom.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=50, attention_masks=custom_attention_masks)
    decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Decoded Output: {decoded_output}")

    print("Main function completed.")

if __name__ == '__main__':
    print("Executing script...")
    main()
    print("Script execution completed.")

# from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Model, GPT2Config
# import torch
# import numpy as np
# import argparse

# print("Starting script...")

# model_path = "gpt2"
# access_token = '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'
# print("Model path and access token set.")

# '''
# Creates attention mask for all layers such that all words whose sum of alpha values is less than threshold are dropped out.

# Params:
# attentions (Tuple of tensors) - Attention matrix that model outputs
# threshold - threshold under which elements are dropped out

# Returns tensor of size [num_layers, batch_size, num_heads, seq_length, seq_length]
# '''
# def create_attention_mask(attentions, threshold):
#     print("Creating attention masks...")
#     batch_size, num_heads, seq_length = attentions[0].size(0), attentions[0].size(1), attentions[0].size(2)
#     attentions_np = np.stack([attn.detach().cpu().numpy() for attn in attentions])
#     sums = np.sum(attentions_np, axis=3)
#     masks = np.where(sums < threshold, 0, 1)
#     attention_masks = np.repeat(masks[:, :, :, np.newaxis], seq_length, axis=3)
#     print("Attention masks created.")
#     return torch.from_numpy(attention_masks).to(attentions[0].device)


# class CustomGPT2Model(GPT2Model):
#     def __init__(self, config):
#         print("Initializing CustomGPT2Model...")
#         super(CustomGPT2Model, self).__init__(config)
#         print("CustomGPT2Model initialized.")
    
#     def forward(self, attention_masks, inputs_embeds=None, **kwargs):
#         print("Starting forward pass in CustomGPT2Model...")
#         all_hidden_states = []
#         hidden_states = inputs_embeds
        
#         for i, layer_module in enumerate(self.h):
#             print(f"Processing layer {i}...")
#             if attention_masks is not None and i < len(attention_masks):
#                 attention_mask = attention_masks[i]
#                 print(f"Applied attention mask for layer {i}.")
#             else:
#                 attention_mask = None
#             #make a forward pass through the layer

#             layer_outputs = layer_module(
#                 hidden_states,
#                 attention_mask=attention_mask,
#                 **kwargs
#             )
#             hidden_states = layer_outputs[0]
#             all_hidden_states.append(hidden_states)
#             print(f"Layer {i} processed.")
        
#         hidden_states = self.ln_f(hidden_states)
#         all_hidden_states.append(hidden_states)
#         print("Forward pass in CustomGPT2Model completed.")
#         return hidden_states, all_hidden_states

# print("Setting up argument parser...")
# parser = argparse.ArgumentParser(description='Choose whether to finetune or evaluate.')
# parser.add_argument('threshold', metavar='threshold', type=float, nargs='?', default=0.5,
#                     help='Threshold t such that, if sum(alpha_w) > t, for given w, model is said to attend to w')
# parser.add_argument('finetune', metavar='finetune', type=bool, nargs='?', default=False, help='Whether to finetune model or not')
# print("Argument parser set up.")

# print("Loading tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained(model_path, force_download=True, token=access_token)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"
# print("Tokenizer loaded.")

# print("Loading model...")
# model = AutoModelForCausalLM.from_pretrained(model_path, output_attentions=True, token=access_token)
# print("Model loaded.")

# args = parser.parse_args()
# print(f"Arguments parsed: {args}")

# def main():
#     print("Entering main function...")
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     elif torch.backends.mps.is_available():
#         device = torch.device("mps")
#     else:
#         device = torch.device("cpu")
#     print(f"Using device: {device}")

#     prompt = "What does this code do?"
#     print(f"Tokenizing prompt: {prompt}")
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=model.config.n_positions).to(device)
#     print(f"Inputs: {inputs}")

#     print("Moving model to device...")
#     model.to(device)
#     print("Generating model outputs...")
#     outputs = model(**inputs)
#     attentions = outputs.attentions
#     print(f"Model outputs generated. Attentions: {attentions}")

#     print("Creating attention masks...")
#     attention_masks = create_attention_mask(attentions, args.threshold)
#     print(f"Attention masks: {attention_masks}")

#     input_ids = inputs['input_ids']
#     print(f"Generating embeddings for input IDs: {input_ids}")
#     embeddings = model.get_input_embeddings()(input_ids)
#     print(f"Embeddings: {embeddings}")

#     print("Loading custom model configuration...")
#     config = GPT2Config.from_pretrained(model_path)
#     model_custom = CustomGPT2Model.from_pretrained(model_path, config=config).to(device)
#     print("Custom model loaded and moved to device.")

#     print("Running custom forward pass...")
#     with torch.no_grad():
#         final_output, all_hidden_states = model_custom(
#             inputs_embeds=embeddings,
#             attention_masks=attention_masks
#         )
#     print("Custom forward pass completed.")

#     # Store results/tensors of the second forward pass
#     model_name = model_path.replace('/', '_')
#     masked_forward_results = {f'layer_{i}': hidden_state for i, hidden_state in enumerate(all_hidden_states)}
#     save_path = f'{model_name}_masked_forward_results.pth'
#     print(f"Storing masked forward results to {save_path}...")
#     torch.save(masked_forward_results, save_path)
#     print(f"Masked forward results saved to {save_path}.")

#     print("Final Model Output:")
#     print(final_output)
#     #decode final output
#     index = torch.argmax(final_output[0])
#     print(tokenizer.decode(final_output[0]))
#     print("All Hidden States:")
#     for i, hidden_state in enumerate(all_hidden_states):
#         print(f"Layer {i} Hidden State Shape:", hidden_state.shape)
#     print("Main function completed.")

# if __name__ == '__main__':
#     print("Executing script...")
#     main()
#     print("Script execution completed.")
