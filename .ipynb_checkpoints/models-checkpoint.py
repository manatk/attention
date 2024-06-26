import torch
import torch.nn as nn
from torch.nn import functional as F
import math
torch.manual_seed(0)
import pdb

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = config.n_head
        self.head_dim = config.n_embd // self.num_heads
        self.all_head_dim = self.num_heads * self.head_dim

        self.query = nn.Linear(config.n_embd, self.all_head_dim)
        self.key = nn.Linear(config.n_embd, self.all_head_dim)
        self.value = nn.Linear(config.n_embd, self.all_head_dim)

        self.out = nn.Linear(self.all_head_dim, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.proj_dropout = nn.Dropout(config.resid_pdrop) 

    def forward(self, hidden_states, attention_mask=None):
        pdb.set_trace()
        batch_size, seq_length, hidden_size = hidden_states.size()

        # Linear projections to obtain Q, K, V
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        # Reshape and transpose for multi-head attention
        query_layer = query_layer.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_layer = key_layer.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value_layer = value_layer.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attention_scores *= attention_mask  # Apply attention mask here

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(1, 2).contiguous().view(batch_size, seq_length, self.all_head_dim)

        # Final linear projection
        output = self.out(context_layer)
        output = self.proj_dropout(output)

        return output

class GPTConfig:
    """Base GPT config, params common to all GPT versions"""
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, n_embd=768, n_layer=12, n_head=12):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head

class GPT1Config(GPTConfig):
    """GPT-1 like network roughly 125M params"""
    pass

class Block(nn.Module):
    """An unassuming Transformer block"""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadSelfAttention(config)
        self.n_attn_layers = len(config.attention_masks)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        self.attention_masks = config.attention_masks

    def forward(self, x):
        for idx, attention_mask in enumerate(self.attention_masks):
            x = x + self.attn(self.ln1(x), attention_mask=attention_mask)  # Pass attention mask to self.attn
        x = x + self.mlp(self.ln2(x))
        return x

class CustomMultiHeadAttentionGPT(nn.Module):
    """The full GPT language model, with a context size of block_size"""

    def __init__(self, config):
        super().__init__()

        # Input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer
        self.blocks = [Block(config) for _ in range(config.n_layer)]

        # Decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.attention_masks = config.attention_masks  # Ensure attention_masks is stored

        print(f"number of parameters: {sum(p.numel() for p in self.parameters())}")

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, attention_masks=None, targets=None):
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward, model block size ({t}, {self.block_size}) is exhausted."

        # Forward the GPT model
        token_embeddings = self.tok_emb(idx)  # Each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :]  # Each position maps to a (learnable) vector
        x_input = token_embeddings + position_embeddings

        x = self.drop(x_input)
        for block in self.blocks:
            x = block(x)
        '''for i, block in enumerate(self.blocks):
            pdb.set_trace()
            x = block(x, attention_mask=self.attention_masks[i] if attention_masks is not None else None)  # Pass the specific mask
            pdb.set_trace()'''
        x = self.ln_f(x)
        logits = self.head(x)

        # If we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)

        return logits, loss

    def generate(self, input_ids, attention_masks=None, max_length=50):
        generated = input_ids
        for _ in range(max_length - input_ids.size(1)):
            logits, _ = self.forward(generated, attention_masks=attention_masks)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            generated = torch.cat((generated, next_token), dim=1)
        return generated
"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier


Originally forked from Andrej Karpathy's minGPT.

CS224N 2023-24: Homework 4

John Hewitt <johnhew@stanford.edu>
Ansh Khurana <anshk@stanford.edu>
Soumya Chatterjee <soumyac@stanford.edu>
"""
'''
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

torch.manual_seed(0)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = config.n_head
        self.head_dim = config.n_embd // self.num_heads
        self.all_head_dim = self.num_heads * self.head_dim

        self.query = nn.Linear(config.n_embd, self.all_head_dim)
        self.key = nn.Linear(config.n_embd, self.all_head_dim)
        self.value = nn.Linear(config.n_embd, self.all_head_dim)

        self.out = nn.Linear(self.all_head_dim, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.proj_dropout = nn.Dropout(config.resid_pdrop)


    def forward(self, hidden_states, attention_masks=None, **kwargs):
        batch_size, seq_length, hidden_size = hidden_states.size()

        # Linear projections to obtain Q, K, V
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        # Reshape and transpose for multi-head attention
        query_layer = query_layer.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_layer = key_layer.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value_layer = value_layer.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_masks is not None:
            attention_scores = attention_scores * attention_masks

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(1, 2).contiguous().view(batch_size, seq_length, self.all_head_dim)

        # Final linear projection
        output = self.out(context_layer)
        output = self.proj_dropout(output)

        return output



class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    rope = False
    bottleneck_dim = None

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class CustomMultiHeadAttentionGPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        if not config.rope:
            self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.rope = config.rope
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        print(f"number of parameters: {sum(p.numel() for p in self.parameters())}")

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward, model block size ({t}, {self.block_size}) is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        if self.rope:
            x_input = token_embeddings
        else:
            position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
            x_input = token_embeddings + position_embeddings

        x = self.drop(x_input)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)

        return logits, loss
'''