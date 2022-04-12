import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# device = 'cuda:1'

cache_dir = "/data4/yoomcache"
model_cache_dir = os.path.join(cache_dir, 'huggingface')
data_cache_dir = os.path.join(cache_dir, 'datasets')
checkpoint_dir = os.path.join(cache_dir, 'checkpoint')

import torch
import torch.nn as nn

import warnings


from huggingface.modeling_wav2vec2 import *
from huggingface.modeling_gpt2 import *
from huggingface.modeling_gpt2 import GPT2_START_DOCSTRING, GPT2_INPUTS_DOCSTRING, PARALLELIZE_DOCSTRING, DEPARALLELIZE_DOCSTRING
from configuration_wav2vec2gpt import Wav2Vec2GPTConfig



_HIDDEN_STATES_START_POSITION = 2

# General docstring
_CONFIG_FOR_DOC = "Wav2Vec2GPTConfig" # wav2vec: "Wav2Vec2Config"
_PROCESSOR_FOR_DOC = "Wav2Vec2GPTProcessor" # wav2vec: "Wav2Vec2Processor"

# Base docstring
_CHECKPOINT_FOR_DOC = "wav2vec2gpt-base" # wav2vec: "facebook/wav2vec2-base-960h"
_INPUT_AUDIO_DIR = ""
_EXPECTED_OUTPUT_SHAPE = [1, 1024, 50257] # wav2vec: [1, 292, 768]
_EXPECTED_OUTPUT = ""
_CTC_EXPECTED_LOSS = 100000






@add_start_docstrings(
    "The bare Wav2Vec2 Model transformer outputting raw hidden-states without any specific head on top.",
    WAV_2_VEC_2_START_DOCSTRING,
)
class Wav2Vec2Model2(Wav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.config = config
        self.feature_extractor = Wav2Vec2FeatureEncoder(config)
        self.feature_projection = Wav2Vec2FeatureProjection(config)

        self.adapter = Wav2Vec2Adapter(config) if config.add_adapter else None

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5."
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.feature_extractor._freeze_parameters()


    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_PROCESSOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Wav2Vec2BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_values,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        hidden_states, extract_features = self.feature_projection(extract_features)

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, extract_features)

        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
        )






@add_start_docstrings(
    "The basic GPT2 Model transformer that starts from `inputs_embeds` not `input_ids`.",
    GPT2_START_DOCSTRING,
)
class GPT2fromEmbedding(GPT2Model):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config):
        super().__init__(config)

        # self.embed_dim = config.hidden_size

        # self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # self.wte = self.wte.to(self.first_device)
        self.wpe = self.wpe.to(self.first_device)
        # Load onto devices
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.h[block] = self.h[block].to(cuda_device)
        # ln_f to last
        self.ln_f = self.ln_f.to(self.last_device)

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        # self.wte = self.wte.to("cpu")
        self.wpe = self.wpe.to("cpu")
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")
        torch.cuda.empty_cache()

    # def get_input_embeddings(self):
    #     return self.wte
    # 
    # def set_input_embeddings(self, new_embeddings):
    #     self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

#     @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
#     @add_code_sample_docstrings(
#         processor_class=_PROCESSOR_FOR_DOC,
#         checkpoint=_CHECKPOINT_FOR_DOC,
#         output_type=BaseModelOutputWithPastAndCrossAttentions,
#         config_class=_CONFIG_FOR_DOC,
#     )
    def forward(
        self,
        # input_ids=None,
        past_key_values=None,
        attention_mask=None,
        # token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if input_ids is not None and inputs_embeds is not None:
        #     raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # elif input_ids is not None:
        #     input_shape = input_ids.size()
        #     input_ids = input_ids.view(-1, input_shape[-1])
        #     batch_size = input_ids.shape[0]
        # elif inputs_embeds is not None:
        #     input_shape = inputs_embeds.size()[:-1]
        #     batch_size = inputs_embeds.shape[0]
        # else:
        #     raise ValueError("You have to specify either input_ids or inputs_embeds")
        if inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify inputs_embeds")

        # device = input_ids.device if input_ids is not None else inputs_embeds.device
        device = inputs_embeds.device

        # if token_type_ids is not None:
        #     token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # # If a 2D or 3D attention mask is provided for the cross-attention
        # # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        # if self.config.add_cross_attention and encoder_hidden_states is not None:
        #     encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        #     encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        #     if encoder_attention_mask is None:
        #         encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
        #     encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        # else:
        #     encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        # if inputs_embeds is None:
        #     inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # if token_type_ids is not None:
        #     token_type_embeds = self.wte(token_type_ids)
        #     hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`transformers."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )



@add_start_docstrings(
    """
    The GPT2 Model transformer that starts with `inputs_embeds` with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    GPT2_START_DOCSTRING,
)
class GPT2LMfromEmbedding(GPT2LMHeadModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        # self.transformer = GPT2Model(config)
        self.transformer = GPT2fromEmbedding(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

#     @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
#     @add_code_sample_docstrings(
#         processor_class=_PROCESSOR_FOR_DOC,
#         checkpoint=_CHECKPOINT_FOR_DOC,
#         output_type=CausalLMOutputWithCrossAttentions,
#         config_class=_CONFIG_FOR_DOC,
#     )
    def forward(
        self,
        # input_ids=None,
        past_key_values=None,
        attention_mask=None,
        # token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        # encoder_hidden_states=None,
        # encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, transformers., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, transformers., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            # input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            # encoder_hidden_states=encoder_hidden_states,
            # encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )




@add_start_docstrings(
    """Wav2Vec2 Model that uses GPT Model as an decoding `language modeling` head on top for Connectionist Temporal Classification (CTC).""",
    WAV_2_VEC_2_START_DOCSTRING,
)
class Wav2Vec2GPTModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Wav2Vec2GPTModel.from_pretrained(transformers., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        
        self.wav2vec2 = Wav2Vec2Model2(config)
        self.dropout = nn.Dropout(config.final_dropout)
        
        self.rnn_compressor = nn.GRU(input_size=config.hidden_size, hidden_size=(config.hidden_size + 1), num_layers=1, batch_first=True)
        self.adaPool = nn.AdaptiveMaxPool1d(config.n_positions, return_indices=True)
        self.n_hidden = config.hidden_size


        self.gpt2lm = GPT2LMfromEmbedding(config)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()
    
    def freeze_feature_projection(self):
        for param in self.wav2vec2.feature_projection.parameters():
            param.requires_grad = False
    
    def freeze_wav2vec_encoder(self):
        for param in self.wav2vec2.encoder.parameters():
            param.requires_grad = False
    
    def freeze_wav2vec_adapter(self):
        for param in self.wav2vec2.adapter.parameters():
            param.requires_grad = False
    
    def freeze_rnn_compressor(self):
        for param in self.rnn_compressor.parameters():
            param.requires_grad = False
    
    def freeze_gpt_decoder(self):
        for param in self.gpt2lm.transformer.parameters():
            param.requires_grad = False
    
    def freeze_lm_head(self):
        for param in self.gpt2lm.lm_head.parameters():
            param.requires_grad = False

    def unfreeze_feature_extractor(self):
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = True
    
    def unfreeze_feature_projection(self):
        for param in self.wav2vec2.feature_projection.parameters():
            param.requires_grad = True
    
    def unfreeze_wav2vec_encoder(self):
        for param in self.wav2vec2.encoder.parameters():
            param.requires_grad = True
    
    def unfreeze_wav2vec_adapter(self):
        for param in self.wav2vec2.adapter.parameters():
            param.requires_grad = True
    
    def unfreeze_rnn_compressor(self):
        for param in self.rnn_compressor.parameters():
            param.requires_grad = True
    
    def unfreeze_gpt_decoder(self):
        for param in self.gpt2lm.transformer.parameters():
            param.requires_grad = True
    
    def unfreeze_lm_head(self):
        for param in self.gpt2lm.lm_head.parameters():
            param.requires_grad = True

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_PROCESSOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT,
        expected_loss=_CTC_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_values,  # wav input
        attention_mask=None,  # wav attention mask
        output_attentions_wav2vec=None,
        output_hidden_states_wav2vec=None, 
        return_dict_wav2vec=None,

        output_attention_mask=None,  # text attention mask
        output_attentions_gpt2=None,
        output_hidden_states_gpt2=None, 
        return_dict_gpt2=None,
        labels=None,
        use_cache=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, transformers., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, transformers.,
            config.vocab_size - 1]`.
        """

        return_dict_wav2vec = return_dict_wav2vec if return_dict_wav2vec is not None else self.config.use_return_dict
        return_dict_gpt2 = return_dict_gpt2 if return_dict_gpt2 is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions_wav2vec,
            output_hidden_states=output_hidden_states_wav2vec,
            return_dict=return_dict_wav2vec,
        )

        hidden_states = outputs.last_hidden_state
        # wav2vec_extracted_feature = outputs.extract_features
        # wav2vec_hidden_states = outputs.hidden_states if output_hidden_states_wav2vec else None
        # wav2vec_attentions = outputs.attentions if output_attentions_wav2vec else None
        
        hidden_states = self.dropout(hidden_states) # size: (batch_size, seq_len_from_adapter, config.hidden_states=768)

        output_from_RNN, _ = self.rnn_compressor(hidden_states) # batch_first is True. size: (batch_size, seq_len, config.hidden_states + 1)
        word_indices = self.adaPool(output_from_RNN[:,:,-1])[1]
        word_embeddings = torch.gather(output_from_RNN[:,:,:-1], 
                                       1, 
                                       word_indices.unsqueeze(-1).expand(-1,-1,self.n_hidden))
        

        return self.gpt2lm(
            ### input_ids=None,
            # past_key_values=None,
            attention_mask=output_attention_mask,
            ### token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=word_embeddings,
            ### encoder_hidden_states=None,
            ### encoder_attention_mask=None,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions_gpt2,
            output_hidden_states=output_hidden_states_gpt2,
            return_dict=return_dict_gpt2,
        ) # Type: CausalLMOutputWithCrossAttentions

