from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaModel, LlamaTokenizer, LlamaConfig
from transformers import LongformerModel, LongformerTokenizer
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datasets import load_dataset
#maybe install sparseAttention package from Microsoft???

#model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
model_path = "allenai/longformer-base-4096"
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
Creates attention mask for all layers such that all words whose sum of alpha values is less than threshold are dropped out.

Params:
attentions (Tuple of tensors) - Attention matrix that model outputs
threshold - threshold under which elements are dropped out
'''
def create_attention_mask(attentions, threshold):
    batch_size, num_heads, seq_length = attentions[0].size()[0], attentions[0].size()[1], attentions[0].size()[2]
    attentions = np.stack([attentions[l].detach().numpy() for l in len(attentions)])
    sums = np.sum(attentions, axis=3)  # Sum over the last axis to combine the attention over sequence length
    sums = np.where(sums < threshold, 0, 1)  # Apply threshold and zero out the lower values: should be Size(batch_size, num_heads, seq_length)
    return np.repeat(sums[:,:,:,:,np.newaxis], seq_length, 3)
    
'''
Creates attention plot for a given layer.

Params:
attentions (Tuple of tensors) - Attention matrix that model outputs.
layer (int) - layer to choose
threshold (float) - threshold for alpha value over which a word is said to be attended to
'''
def plot_attention(attentions, layer, threshold, iter, dataset_name):
    # Iterate through each batch, head, and sequence
    batch_size, num_heads, seq_length = attentions[0].size()[0], attentions[0].size()[1], attentions[0].size()[2]
    attention_layer = attentions[layer].detach().numpy()
    sums = np.sum(attention_layer, axis=2)  # Sum over the last axis to combine the attention over sequence length
    sums = np.where(sums < threshold, 0, sums)  # Apply threshold and zero out the lower values: should be Size(batch_size, num_heads, seq_length)

    # Count non-zero values across batch and head dimensions
    counts = np.count_nonzero(sums, axis=(0, 1))

    x_values = [i for i in range(seq_length)]
    y_values = counts

    # Plot the points
    plt.clf()
    plt.scatter(x_values, y_values)
    plt.xlabel('Position in Sequence')
    plt.ylabel('Count of Words with Attention > ' + str(threshold))
    plt.title('Attention Visualization')
    plt.grid(True)
    plt.savefig('attention_plots/' + dataset_name + 'layer=' + str(layer) + 'iter' + str(iter) + ',' +  'threshold=' + str(threshold) + '.png')
    #plt.show()

# Functions for Book Corpus

# TODO: DO we keep this?
'''
Runs several iterations of training on the same data. Prints first layer of attention for each iteration.

Params:
train_split - Training data
tokenizer - tokenizer
model - model
args - args given by user
'''
def plotDiffIterationsBookCorpus(train_split, tokenizer, model, args, dataset_name):
    for i in range(20):
        context = ""
        for j in range(20):
            context += train_split[i * 20 + j]['text']  # Adjust index to avoid repeats and use different sections
        prompt = context + " Summarize this text."
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=model.config.max_position_embeddings)
        outputs = model(**inputs)
        attentions = outputs.attentions  # Size (num_layers, batch_size, num_heads, sequence_length, sequence_length)
        plot_attention(attentions, layer=0, threshold=args.threshold, iter=i, dataset_name="BooksCorpus") # plot attention

'''
Trains model on training data, with task being to summarize text. Then plots attention matrix for each layer.

Parameters:
train_split - training data, stored as 2D array
tokenizer - tokenizer
model - model
args - args given by user

'''
def plotDifferentLayersBookCorpus(train_split, tokenizer, model, args):
        context = ""    
        for j in range(20):
            context += train_split[j]['text']  # Adjust index to avoid repeats and use different sections
        prompt = context + " Summarize this text."
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=model.config.max_position_embeddings)
        outputs = model(**inputs)
        attentions = outputs.attentions  # Size (num_layers, batch_size, num_heads, sequence_length, sequence_length)
        for i in range(len(attentions)):
            plot_attention(attentions, layer=i, threshold=args.threshold, iter=0, dataset_name="BooksCorpus") # plot attention

# Functions for Stack

# TODO: Do we keep this function?
def plotDiffIterationsStack(train_split, tokenizer, model, args):
        the_stack = load_dataset("bigcode/the-stack", data_dir="data/python", streaming=True, split="train")
        code = ""
        count = 0
        for sample in the_stack:
            #print(sample["content"])
            code += sample["content"]
            prompt = code + " Summarize this text."
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=model.config.max_position_embeddings)
            outputs = model(**inputs)
            attentions = outputs.attentions  # Size (num_layers, batch_size, num_heads, sequence_length, sequence_length)
            for i in range(len(attentions)):
                plot_attention(attentions, layer=i, threshold=args.threshold, iter=i, dataset_name="the_stack") # plot attention  
            count = count + 1
            if count > 20:
                break
        
#TODO: train_split doesn't even appear in this function? You should load the training data in the main file.
'''
Trains model on training data, with task being to explain function of code.  Then plots attention matrix for each layer.

Parameters:
train_split - training data, stored as 2D array
tokenizer - tokenizer
model - model
args - args given by user
'''
def plotDifferentLayersStack(train_split, tokenizer, model, args):
    the_stack = load_dataset("bigcode/the-stack", data_dir="data/python", streaming=True, split="train")
    code = ""
    for sample in the_stack:
        #print(sample["content"])
        code += sample["content"]
        break
    prompt = code + " What does this code do?"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=model.config.max_position_embeddings)
    outputs = model(**inputs)
    attentions = outputs.attentions  # Size (num_layers, batch_size, num_heads, sequence_length, sequence_length)
    for i in range(len(attentions)):
        plot_attention(attentions, layer=i, threshold=args.threshold, iter=i, dataset_name="the_stack") # plot attention for first layer


def main():
    '''
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
    input_text = "test text for input."
    inputs = tokenizer(input_text, return_tensors='pt', max_length=4096, truncation=True, padding=True)
    attention_mask = torch.ones(inputs['input_ids'].shape, dtype=torch.long)
    global_attention_mask = torch.zeros(inputs['input_ids'].shape, dtype=torch.long)
    global_attention_mask[:, 0] = 1
    outputs = model(input_ids=inputs['input_ids'],
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,
                    output_attentions=True)
    attentions = outputs.attentions

    # Print the attention matrices
    plot_attention(attentions, 0, .5, 0, "testSparse")
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_path, force_download=True, token=access_token)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set if the model expects it
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(model_path, output_attentions=True, token=access_token)

    args = parser.parse_args()

    bookcorpus = load_dataset('bookcorpus')
    train_split = bookcorpus["train"]
    # plotDifferentLayersStack(train_split, tokenizer, model, args)
    #plotDiffIterationsBookCorpus(train_split, tokenizer, model, args, "bookCorpus")
    plotDifferentLayersBookCorpus(train_split, tokenizer, model, args)
    # Iterate over the dataset 20 times


    '''
    if 'logits' in outputs:
        decoded_answers = tokenizer.batch_decode(torch.argmax(outputs.logits, dim=-1), skip_special_tokens=True)
    else:
        decoded_answers = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)
    '''
if __name__ == '__main__':
    main()
