def create_attention_mask(attentions, threshold):
    print("Creating attention masks...")
    batch_size, num_heads, seq_length = attentions[0].size(0), attentions[0].size(1), attentions[0].size(2)
    attentions_np = np.stack([attn.detach().cpu().numpy() for attn in attentions])
    sums = np.sum(attentions_np, axis=3)
    masks = np.where(sums < 1 - threshold * seq_length, 0, 1)
    attention_masks = np.repeat(masks[:, :, :, np.newaxis], seq_length, axis=3)
    print("Attention masks created.")
    return torch.from_numpy(attention_masks).to(attentions[0].device)
