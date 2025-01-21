def compute_ctc_loss(criterion, model_output, label, num_items_in_batch=None):
    # model_output[0] : logits (N, T, C)
    # model_output[1] : predicted_ids (N, T)
    # model_output[2] : attention_lengths (N)
    # label['token_ids_asr'] : (N, S_asr)
    # label['attn_mask_asr'] : (N, S_asr)
    # label['token_ids_llm'] : (N, S_llm)
    # label['attn_mask_llm'] : (N, S_llm)
    # num_items_in_batch : add just to handle error in transformers

    log_probs = model_output[0].log_softmax(dim=-1)
    log_probs = log_probs.transpose(0, 1)   # (T, N, C)
    input_lengths = model_output[2]
    targets = label['token_ids_llm']
    target_lengths = label['attn_mask_llm'].sum(dim=-1)
    # print(log_probs.shape, input_lengths.shape, targets.shape, target_lengths.shape)

    # criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    loss = criterion(log_probs, targets, input_lengths, target_lengths)

    return loss


if __name__ == '__main__':
    import torch
    import torch.nn as nn

    batch_size = 7
    max_seq_len = 20

    asr_token_dim = 30
    llm_token_dim = 30
    max_token_len = 10


    # Test
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    logits = torch.randn(batch_size, max_seq_len, llm_token_dim)
    logits_argmax = torch.argmax(logits, dim=-1)
    attn_length = torch.IntTensor([max_seq_len] * batch_size)
    model_output = (logits, logits_argmax, attn_length)

    label = dict(
        token_ids_asr = torch.randint(0, asr_token_dim, (batch_size, max_token_len)),
        attn_mask_asr = torch.ones(batch_size, max_token_len, dtype=torch.int32),
        token_ids_llm = torch.randint(0, llm_token_dim, (batch_size, max_token_len)),
        attn_mask_llm = torch.ones(batch_size, max_token_len, dtype=torch.int32)
    )

    loss = compute_ctc_loss(criterion, model_output, label)
    print(loss.item())