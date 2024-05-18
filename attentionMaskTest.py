from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaModel, LlamaConfig
import torch
import numpy as np
import argparse

model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
access_token = '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'

'''
Creates attention mask for all layers such that all words whose sum of alpha values is less than threshold are dropped out.

Params:
attentions (Tuple of tensors) - Attention matrix that model outputs
threshold - threshold under which elements are dropped out

Returns tensor of size [layer_size, batch_size, num_heads, seq_length, seq_length]
'''
def create_attention_mask(attentions, threshold):
    batch_size, num_heads, seq_length = attentions[0].size(0), attentions[0].size(1), attentions[0].size(2)
    attentions_np = np.stack([attn.detach().cpu().numpy() for attn in attentions])
    sums = np.sum(attentions_np, axis=3)
    masks = np.where(sums < threshold, 0, 1)
    attention_masks = np.repeat(masks[:, :, :, np.newaxis], seq_length, axis=3)
    return torch.from_numpy(attention_masks)

class CustomLlamaModel(LlamaModel):
    def __init__(self, config):
        super(CustomLlamaModel, self).__init__(config)
    
    #THIS IS THE PART I AM STUCK — SPECIFICALLY, HOW DO WE APPLY THE RIGHT MASK TO EACH LAYER AND THEN PASS IT FORWARD
    def forward(self, inputs_embeds=None, attention_masks=None, position_ids=None, **kwargs):
        all_hidden_states = []

        hidden_states = inputs_embeds # What is the size of hidden_states?
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
    
    outputs = model(**inputs)
    attentions = outputs.attentions
    
    attention_masks = create_attention_mask(attentions, args.threshold)
    attention_masks = [torch.tensor(mask, dtype=torch.float32) for mask in attention_masks]
    
    input_ids = inputs['input_ids']
    embeddings = model.get_input_embeddings()(input_ids)

    position_ids = torch.arange(embeddings.size(1), dtype=torch.long)
    position_ids = position_ids.unsqueeze(0).expand(embeddings.size(0), -1)

    config = LlamaConfig.from_pretrained(model_path)
    model_custom = CustomLlamaModel.from_pretrained(model_path, config=config)
    
    # Apply the custom forward pass - NOT SURE IF THIS IS THE RIGHT WAY TO COMBINE ALL TYHE LAYER INPUTS
    final_output, all_hidden_states = model_custom(
        inputs_embeds=embeddings,
        attention_masks=attention_masks,
        position_ids=position_ids
    )
    
    print("Final Model Output:")
    print(final_output)
    print("All Hidden States:")
    for i, hidden_state in enumerate(all_hidden_states):
        print(f"Layer {i} Hidden State Shape:", hidden_state.shape)

if __name__ == '__main__':
    main()
