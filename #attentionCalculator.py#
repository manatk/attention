from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse


# Specify the path to your LLaMA 3 model
model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
access_token = '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'

parser = argparse.ArgumentParser(description='Choose whether to finetune or evaluate.')
parser.add_argument('threshold', metavar='threshold', type=float, nargs='?', default = 0.5,
                    help='Threshold t such that, if sum(alpha_w) > t, for given w, model is said to attend to w')
parser.add_argument('finetune', metavar = 'finetune', type = bool, nargs='?', default = False, help = 'Whether to finetune model or not')

'''
Prints out useful info from the attention matrix.

Params:
attentions (Tuple of Tensors) - Attention matrix that model outputs. Tuple of (num_layers) tensors, each of size (batch_size, num_heads, sequence_length, sequence_length)
'''
def debug(attentions):
    # Print the shape of the attention matrix  
    # Access the attention from the first layer, first batch, and the first head
    first_layer_first_head_attention = attentions[0][0][0]
    print("Shape of the first layer, first head attention matrix:", first_layer_first_head_attention.shape)
    # Print the attention matrix for the first layer, first head
    print("Attention matrix for the first layer, first head:\n", first_layer_first_head_attention)
    print(attentions[0][0][0][0])

'''
For the attention matrix of a given layer, prints sum of alpha values for each word, in other words, the extent to which word is attended to by the sequence.

Params:
attentions (Tuple of tensors) - Attention matrix that model outputs.
layer (int) - layer to choose
'''
def print_attention_sums(attentions, layer):
    # Creating plot for first layer
    # Iterate through each batch, head, and sequence
    batch_size, num_heads, seq_length = attentions[0].size()[0], attentions[0].size()[1], attentions[0].size()[2]
    points = [0 for i in range(seq_length)]
    attention_layer = attentions[layer].detach().numpy()
    for batch in range(batch_size):
        for head in range(num_heads):
            for seq_pos in range(seq_length):
                # Calculate the sum of attention values for the current word
                attention_sum = np.sum(attention_layer[batch, head, :,seq_pos])
                print(attention_sum)
'''
Creates attention plot for a given layer.

Params:
attentions (Tuple of tensors) - Attention matrix that model outputs.
layer (int) - layer to choose
threshold (float) - threshold for alpha value over which a word is said to be attended to
'''
def plot_attention(attentions, layer, threshold):
    # Creating plot for first layer
    # Iterate through each batch, head, and sequence
    batch_size, num_heads, seq_length = attentions[0].size()[0], attentions[0].size()[1], attentions[0].size()[2]
    points = [0 for i in range(seq_length)]
    attention_layer = attentions[layer].detach().numpy()
    sums = np.sum(attention_layer, 2)
    sums = np.putmask(sums, sums<threshold, 0)
    counts = numpy.count_nonzero(sums, [0, 1, 3])
    x_values = [i for i in range(seq_length)]
    y_values = counts
    # Plot the points
    plt.scatter(x_values, y_values)
    plt.xlabel('Position in Sequence')
    plt.ylabel('Count of Words with Attention > ' + str(threshold))
    plt.title('Attention Visualization')
    plt.grid(True)
    plt.savefig('attention_plots/layer=' + str(layer) + ',threshold=' + str(threshold) + '.png')
    plt.show()

def main():
    #pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", token = access_token)
    # reload base model
    # Load the tokenizer and model
    #tokenizer = AutoTokenizer.from_pretrained(model_path, token=access_token)
    tokenizer = AutoTokenizer.from_pretrained(model_path, force_download=True,token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_path, output_attentions=True, token=access_token)
    args = parser.parse_args()
    if args.finetune:
        # TODO: Put in finetuning code
        pass
    # Example text to encode
    input_text = "Here is some example text to encode."
    
    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt")

    # Pass the tokenized input to the model and request attention matrices
    outputs = model(**inputs)
    
    # Extract attention matrices
    attentions = outputs.attentions  # Size (num_layers, batch_size, num_heads, sequence_length, sequence_length)
    plot_attention(attentions, layer=0, threshold=args.threshold) # plot attention for first layer

if __name__ == '__main__':
    main()
