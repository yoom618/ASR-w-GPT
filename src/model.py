
import torch
import torch.nn as nn
from transformers import AutoModel


class Wav2Vec2Mistral(nn.Module):
    def __init__(self, model_asr, model_llm_token_embed, model_llm_positional_embed, llm_input_dim=5120):
        super(Wav2Vec2Mistral, self).__init__()
        self.llm_input_dim = llm_input_dim
        self.model_asr = model_asr                          # Wav2Vec2Model (wav -> 768)
        self.adapter = nn.Linear(768, self.llm_input_dim)   # e.g. (768 -> 5120)

        self.lm_head = nn.Parameter(
            nn.functional.normalize(model_llm_token_embed.weight.data.T, dim=0).contiguous())  # (5120, 131072)
        
        self.pos_embed = model_llm_positional_embed
        
        self._freeze_parameters(self.model_asr.feature_extractor)
        self._freeze_parameters(self.model_asr.feature_projection)
        # self._freeze_parameters(self.model_asr.encoder)
        self._freeze_parameters(self.lm_head)

    def _freeze_parameters(self, model):
        if isinstance(model, nn.Parameter):
            model.requires_grad = False
        elif isinstance(model, nn.Module):
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, audio, audio_attention_mask=None, **kwargs):
        # audio: (batch, time), audio_attention_mask: (batch, time)
        extract_features = self.model_asr.feature_extractor(audio)  # (batch, 512, time_shrinked)
        extract_features = extract_features.transpose(1, 2)         # (batch, time_shrinked, 512)
        
        audio_attention_mask = self.model_asr._get_feature_vector_attention_mask(
            extract_features.shape[1], audio_attention_mask, add_adapter=False
        )   # audio_attention_mask : (batch, time_shrinked)
        audio_attn_length = audio_attention_mask.sum(dim=1).int()   # (batch,)
        # self._print_feature(audio_attn_length)

        hidden_states, _ = self.model_asr.feature_projection(extract_features)  # (batch, time_shrinked, 768)

        encoder_outputs = self.model_asr.encoder(
            hidden_states,
            attention_mask=audio_attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )   
        asr_last_hidden_state = encoder_outputs.last_hidden_state   # (batch, time_shrinked, 768)

        lm_expected_embed = self.adapter(asr_last_hidden_state)      # (batch, time_shrinked, 5120)
        # self._print_feature(lm_expected_embed)


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


    model = Wav2Vec2Mistral(model_asr, model_llm.embed_tokens, model_llm.rotary_emb, llm_input_dim=4096)
    print(model)


    # toy data test
    import torch
    audio = torch.randn(7, 16000)
    audio_mask = torch.ones(7, 16000)
    token_logits, token_predictions, audio_attn_length = model(audio, audio_mask)
    print(token_predictions.shape, audio_attn_length)