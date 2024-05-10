from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Specify the path to your LLaMA 3 model
model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
access_token = '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'
#pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", token = access_token)

# Load the tokenizer and model
#tokenizer = AutoTokenizer.from_pretrained(model_path, token=access_token)
tokenizer = AutoTokenizer.from_pretrained(model_path, force_download=True,token=access_token)
model = AutoModelForCausalLM.from_pretrained(model_path, output_attentions=True, token=access_token)

# Example text to encode
input_text = "Here is some example text to encode."

# Tokenize the input
inputs = tokenizer(input_text, return_tensors="pt")

# Pass the tokenized input to the model and request attention matrices
outputs = model(**inputs)

# Extract attention matrices
attentions = outputs.attentions  # This is a tuple of tensors

# Access the attention from the first layer and the first head
first_layer_first_head_attention = attentions[0][0]

# Print the shape of the attention matrix
print("Shape of the first layer, first head attention matrix:", first_layer_first_head_attention.shape)

# Print the attention matrix for the first layer, first head
print("Attention matrix for the first layer, first head:\n", first_layer_first_head_attention)
