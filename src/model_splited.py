
from typing import Union
import torch
import torch.nn as nn
from transformers import AutoModel


class Wav2Vec2Embedding(nn.Module):
    def __init__(self, model_asr):
        super().__init__()
        self.feature_extractor = model_asr.feature_extractor
        self.feature_projection = model_asr.feature_projection
        self.encoder = model_asr.encoder

        self.output_dim = model_asr.config.hidden_size
        self.conv_kernel = model_asr.config.conv_kernel
        self.conv_stride = model_asr.config.conv_stride
        
        self._freeze_parameters(self.feature_extractor)
        self._freeze_parameters(self.feature_projection)

    def _freeze_parameters(self, model):
        if isinstance(model, nn.Parameter):
            model.requires_grad = False
        elif isinstance(model, nn.Module):
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, audio, audio_attention_mask=None, **kwargs):
        # audio: (batch, time), audio_attention_mask: (batch, time)
        extract_features = self.feature_extractor(audio)  # (batch, 512, time_shrinked)
        extract_features = extract_features.transpose(1, 2)         # (batch, time_shrinked, 512)
        
        audio_attention_mask = self._get_feature_vector_attention_mask(
            extract_features.shape[1], audio_attention_mask
        )   # audio_attention_mask : (batch, time_shrinked)
        audio_attn_length = audio_attention_mask.sum(dim=1).int()   # (batch,)
        # self._print_feature(audio_attn_length)

        hidden_states, _ = self.feature_projection(extract_features)  # (batch, time_shrinked, 768)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=audio_attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )   
        asr_last_hidden_state = encoder_outputs.last_hidden_state   # (batch, time_shrinked, 768)

        return asr_last_hidden_state, audio_attn_length
    
    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor):

        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths)
        output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(self.conv_kernel, self.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
        

        return input_lengths


class Emb2Emb_adapter(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.adapter = nn.Linear(input_dim, output_dim)

    def forward(self, hidden_states):
        return self.adapter(hidden_states)
    


class Emb2Mistral(nn.Module):
    def __init__(self, model_llm):
        super().__init__()
        self.llm_input_dim = model_llm.config.hidden_size

        self.lm_head = nn.Parameter(
            nn.functional.normalize(model_llm.embed_tokens.weight.data.T, dim=0).contiguous())  # (5120, 131072)
        
        self.pos_embed = model_llm.rotary_emb
        
        self._freeze_parameters(self.lm_head)

    def _freeze_parameters(self, model):
        if isinstance(model, nn.Parameter):
            model.requires_grad = False
        elif isinstance(model, nn.Module):
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, lm_expected_embed, audio_attn_length):
        
        past_seen_tokens = 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + lm_expected_embed.shape[1], device=lm_expected_embed.device
        )
        position_ids = cache_position.unsqueeze(0)  # (1, time_shrinked)
        # self._print_feature(position_ids)

        # RoPE 회전을 역으로 되돌리기 위해 -position_ids 사용
        # attention head가 32개, key-value head가 8개이므로 (32+8) * 128 = 5120이 되는 것
        cos, sin = self.pos_embed(lm_expected_embed, -position_ids) # (1, time_shrinked, 128)
        n_head = self.llm_input_dim // 128
        cos = cos.repeat(lm_expected_embed.shape[0], 1, n_head)         # (batch, time_shrinked, 5120)
        sin = sin.repeat(lm_expected_embed.shape[0], 1, n_head)         # (batch, time_shrinked, 5120)
        lm_embed_before_rope = lm_expected_embed * cos + (self.rotate_half(lm_expected_embed) * sin)
        # self._print_feature(lm_embed_before_rope)

        # compute similarity
        similarity_logits = torch.matmul(lm_embed_before_rope, self.lm_head)   # (batch, time_shrinked, 131072)
        token_predictions = similarity_logits.argmax(dim=-1)                   # (batch, time_shrinked)
        # self._print_feature(token_predictions)

        return similarity_logits, token_predictions, audio_attn_length
    
    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)




class Wav2Vec2Mistral(nn.Module):
    def __init__(self, model_asr, model_llm):
        super(Wav2Vec2Mistral, self).__init__()

        self.asr2emb = Wav2Vec2Embedding(model_asr)
        self.emb2emb = Emb2Emb_adapter(model_asr.config.hidden_size, model_llm.config.hidden_size)
        self.emb2lm = Emb2Mistral(model_llm)

    def forward(self, audio, audio_attention_mask=None, **kwargs):
        asr_last_hidden_state, audio_attn_length = self.asr2emb(audio, audio_attention_mask)
        # self._print_feature(asr_last_hidden_state)

        lm_expected_embed = self.emb2emb(asr_last_hidden_state)
        # self._print_feature(lm_expected_embed)

        token_logits, token_predictions, audio_attn_length = self.emb2lm(lm_expected_embed, audio_attn_length)
        # self._print_feature(token_logits)
        # self._print_feature(token_predictions)

        return token_logits, token_predictions, audio_attn_length


    def _print_feature(self, feature):
        import inspect
        callers_local_vars = inspect.currentframe().f_back.f_locals.items()
        get_variable_name = lambda var: [name for name, value in callers_local_vars if value is var][0]
        print(f'{get_variable_name(feature)}:', feature.shape, feature, sep='\n', end='\n\n')




if __name__ == "__main__":
    from utils import set_huggingface_cache_dir

    cache_dir = "/data/yoom618/datasets/"
    token = set_huggingface_cache_dir(cache_dir)
    
    # asr_model_name = "facebook/wav2vec2-base-960h"
    asr_model_name = "facebook/wav2vec2-base"
    # asr_model_name = "openai/whisper-small"

    # llm_model_name = "openai-community/gpt2"
    # llm_model_name = "mistralai/Mistral-Nemo-Instruct-2407"
    llm_model_name = "mistralai/Mistral-7B-v0.1"


    model_asr = AutoModel.from_pretrained(asr_model_name,
                                          cache_dir=cache_dir,
                                          token=token)
    print(f"ASR model: {asr_model_name}")
    print(model_asr)
    print()

    model_llm = AutoModel.from_pretrained(llm_model_name,
                                          cache_dir=cache_dir,
                                          token=token)
    print(f"LLM model: {llm_model_name}")
    print(model_llm)
    print()


    model = Wav2Vec2Mistral(model_asr, model_llm)
    print(model)


    # toy data test
    import torch
    audio = torch.randn(7, 16000)
    audio_mask = torch.ones(7, 16000)
    token_logits, token_predictions, audio_attn_length = model(audio, audio_mask)
    print(token_predictions.shape, audio_attn_length)