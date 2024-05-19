from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaModel, LlamaConfig
import torch
import numpy as np
import argparse

model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
access_token = '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'

'''
Params: This function takes in the attention masks which are outputted by the first forward pass of the model and a threshold value.
Returns: A list of attention masks for each layer of the model.

This function is called after we make our initial call to the model and get the attentions that it is using. We will retain as few
tokens as required such that the sum of the attention weights >= the threshold value
'''

def create_attention_mask(attentions, threshold):
    batch_size, num_heads, seq_length = attentions[0].size(0), attentions[0].size(1), attentions[0].size(2)
    attentions_np = np.stack([attn.detach().cpu().numpy() for attn in attentions])
    sums = np.sum(attentions_np, axis=3)
    masks = np.where(sums < threshold, 0, 1)
    attention_masks = np.repeat(masks[:, :, :, np.newaxis], seq_length, axis=3)
    return attention_masks


class CustomLlamaModel(LlamaModel):
    def __init__(self, config):
        super(CustomLlamaModel, self).__init__(config)
    
    #this function is supposed 
    def forward(self, attention_masks, position_ids=None, inputs_embeds=None, **kwargs):
        # write me code for a forward function which mirrors the forward pass of the llama model except that we for each layer, we use the mask is given by attention_masks[i]
        # and we use the input embeddings given by inputs_embeds
        # we also need to pass the position_ids to the forward pass
        # we need to return the final output and all the hidden states
        all_hidden_states = []

        hidden_states = inputs_embeds
        for i, layer_module in enumerate(self.layers):
            if attention_masks is not None and i < len(attention_masks):
                attention_mask = attention_masks[i].sum(dim=1).unsqueeze(-1).to(hidden_states.device)
                hidden_states = hidden_states * attention_mask

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs
            )
            hidden_states = layer_outputs[0]

            all_hidden_states.append(hidden_states)

        hidden_states = self.norm(hidden_states)
        all_hidden_states.append(hidden_states)
        return hidden_states, all_hidden_states

        '''
        hidden_states = inputs_embeds
        all_hidden_states = []
        for i, layer in enumerate(self.layers):
            layer_output = layer(
                attention_mask=attention_masks[i],
                position_ids=position_ids,
                hidden_states=hidden_states,
                **kwargs
            )
            all_hidden_states.append(layer_output)
            #feed the output of the current layer to the next layer
            hidden_states = layer_output[0]
        #return the final decoder output and all the hidden states
        return layer_output, all_hidden_states
        '''

#parse the arguments
parser = argparse.ArgumentParser(description='Choose whether to finetune or evaluate.')
parser.add_argument('threshold', metavar='threshold', type=float, nargs='?', default=0.5,
                    help='Threshold t such that, if sum(alpha_w) > t, for given w, model is said to attend to w')
parser.add_argument('finetune', metavar='finetune', type=bool, nargs='?', default=False, help='Whether to finetune model or not')

tokenizer = AutoTokenizer.from_pretrained(model_path, force_download=True, token=access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(model_path, output_attentions=True, token=access_token)
args = parser.parse_args()

def main():
    prompt = "What does this code do?"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=model.config.max_position_embeddings)
    
    #This is step 1 - generate the output of the model and see what it is attending to
    outputs = model(**inputs)
    attentions = outputs.attentions
    print(attentions)

    #This is step 2 - create the attention masks
    attention_masks = create_attention_mask(attentions, args.threshold)
    attention_masks = [torch.tensor(mask, dtype=torch.float32) for mask in attention_masks]
    
    #In step 3, we will pass the attention masks to the model and see what it outputs and how that performs
    #In order to do a custom pass of the model, we need to access the input embeddings
    input_ids = inputs['input_ids']
    embeddings = model.get_input_embeddings()(input_ids)

    # position_ids = torch.arange(embeddings.size(1), dtype=torch.long)
    # position_ids = position_ids.unsqueeze(0).expand(embeddings.size(0), -1)

    #We are creating a LLama3 model with a custom forward pass such that it applies the attention masks
    config = LlamaConfig.from_pretrained(model_path)
    model_custom = CustomLlamaModel.from_pretrained(model_path, config=config)
    
    #apply the custom forward pass. Forward output is what we will benchmark
    final_output, all_hidden_states = model_custom(
        inputs_embeds=embeddings,
        attention_masks=attention_masks
    )
    
    print("Final Model Output:")
    print(final_output)
    print("All Hidden States:")
    for i, hidden_state in enumerate(all_hidden_states):
        print(f"Layer {i} Hidden State Shape:", hidden_state.shape)

if __name__ == '__main__':
    main()
