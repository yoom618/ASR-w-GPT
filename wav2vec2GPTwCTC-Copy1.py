import torch
import torch.nn as nn

import warnings


from huggingface.modeling_wav2vec2 import *
from huggingface.modeling_gpt2 import *
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
        ### self.feature_projection = Wav2Vec2FeatureProjection(config)

        self.adapter = Wav2Vec2Adapter(config) if config.add_adapter else None

        ##### Initialize weights and apply final processing
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
        attention_mask=None, # not used in this model since no Encoder
        mask_time_indices=None, # not used in this model since no Encoder
        output_attentions=None, # not used in this model since no Encoder
        output_hidden_states=None,
        return_dict=None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        ### hidden_states, _ = self.feature_projection(extract_features)

        if self.adapter is not None:
            # hidden_states = self.adapter(hidden_states)
            hidden_states = self.adapter(extract_features)
            
            if not return_dict:
                return (hidden_states, extract_features)

            return Wav2Vec2BaseModelOutput(
                last_hidden_state=hidden_states,
                extract_features=extract_features,
            )
        
        else:
            if not return_dict:
                return (extract_features)

            return Wav2Vec2BaseModelOutput(
                # last_hidden_state=hidden_states,
                extract_features=extract_features,
            )
            



@add_start_docstrings(
    """Wav2Vec2 Model that uses GPT Model as an decoding `language modeling` head on top for Connectionist Temporal Classification (CTC).""",
    WAV_2_VEC_2_START_DOCSTRING,
)
class Wav2Vec2GPTModel(Wav2Vec2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

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
        
#         self.rnn_compressor = nn.GRU(
#             input_size=config.output_hidden_size * (2 ** config.num_adapter_layers), 
            
#             bidirectional=False, hidden_size=config.hidden_size, 
#             num_layers=1, 
#             batch_first=True,
#             bias=True
#         )
        
        self.rnn_compressor = nn.Conv1d(
            in_channels=config.output_hidden_size * (2 ** config.num_adapter_layers), 
            out_channels=config.hidden_size, 
            kernel_size=3, 
            stride=1,
            bias=True,
        )

        self.n_hidden = config.hidden_size
        self.n_positions = config.n_positions

        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()
        
        self.select_random = config.select_random
        self.loss_ver = config.loss_ver

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

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, new_embeddings):
        self.transformer.wte = new_embeddings

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

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor.eval()
        self.wav2vec2.feature_extractor._freeze_parameters()
    
    def freeze_feature_projection(self):
        self.wav2vec2.feature_projection.eval()
        for param in self.wav2vec2.feature_projection.parameters():
            param.requires_grad = False
    
    def freeze_wav2vec_encoder(self):
        self.wav2vec2.encoder.eval()
        for param in self.wav2vec2.encoder.parameters():
            param.requires_grad = False
    
    def freeze_wav2vec_adapter(self):
        self.wav2vec2.adapter.eval()
        for param in self.wav2vec2.adapter.parameters():
            param.requires_grad = False
    
    def freeze_rnn_compressor(self):
        self.rnn_compressor.eval()
        for param in self.rnn_compressor.parameters():
            param.requires_grad = False
    
    def freeze_gpt_decoder(self):
        self.transformer.eval()
        for param in self.transformer.parameters():
            param.requires_grad = False
    
    def freeze_lm_head(self):
        self.lm_head.eval()
        for param in self.lm_head.parameters():
            param.requires_grad = False

    def unfreeze_feature_extractor(self):
        self.wav2vec2.feature_extractor.train()
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = True
    
    def unfreeze_feature_projection(self):
        self.wav2vec2.feature_projection.train()
        for param in self.wav2vec2.feature_projection.parameters():
            param.requires_grad = True
    
    def unfreeze_wav2vec_encoder(self):
        self.wav2vec2.encoder.train()
        for param in self.wav2vec2.encoder.parameters():
            param.requires_grad = True
    
    def unfreeze_wav2vec_adapter(self):
        self.wav2vec2.adapter.train()
        for param in self.wav2vec2.adapter.parameters():
            param.requires_grad = True
    
    def unfreeze_rnn_compressor(self):
        self.rnn_compressor.train()
        for param in self.rnn_compressor.parameters():
            param.requires_grad = True
    
    def unfreeze_gpt_decoder(self):
        self.transformer.train()
        for param in self.transformer.parameters():
            param.requires_grad = True
    
    def unfreeze_lm_head(self):
        self.lm_head.train()
        for param in self.lm_head.parameters():
            param.requires_grad = True
    
    
    
    # @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    # @add_code_sample_docstrings(
    #     processor_class=_PROCESSOR_FOR_DOC,
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=CausalLMOutput,
    #     config_class=_CONFIG_FOR_DOC,
    #     expected_output=_EXPECTED_OUTPUT,
    #     expected_loss=_CTC_EXPECTED_LOSS,
    # )
    def forward(
        self,
        input_values,  # wav input
        input_attention_mask=None,  # wav attention mask
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

        ############# Wav2Feature : Referred from Wav2Vec2Model #############

        outputs = self.wav2vec2(
            input_values,
            attention_mask=input_attention_mask,
            # attention_mask=None,
            output_attentions=output_attentions_wav2vec,
            output_hidden_states=output_hidden_states_wav2vec,
            return_dict=return_dict_wav2vec,
        )

        hidden_states_wav = self.dropout(outputs.last_hidden_state)

        
        ############# Feature Propagation : Proposed #############
        
        ##### 1. RNN Module
        # hidden_states_wav, _ = self.rnn_compressor(hidden_states_wav) 
        ##### 2. CNN Module
        hidden_states_wav = self.rnn_compressor(hidden_states_wav.transpose(1,2)).transpose(1,2)
        

        ############# Peak Detection : Proposed #############
        
        pooling_size = 256 if self.loss_ver[:3]=='ctc' else 64
        adaPool = nn.AdaptiveMaxPool1d(pooling_size, return_indices=True)
        
        
        ##### 1.1. cos difference (apply when cosine similarity is between 
        peak_min, threshold, peak_max = -1.0, -0.5, 1.0
        cos = nn.CosineSimilarity(dim=-1, eps=1e-10)
        peak = 1 - cos(hidden_states_wav[...,:-1,:], hidden_states_wav[...,1:,:])
        
        # ##### 1.2. label difference
        # peak_min, threshold, peak_max = 0.0, 0.2, 1.0
        # peak = nn.functional.log_softmax(self.lm_head(hidden_states_wav), dim=-1)
        # peak = - (peak[...,:-1,:] - peak[...,1:,:]).min(dim=-1)[0]
        
        
        # ##### 2.1. using all N tokens
        # pass
        
        ##### 2.2. use only upper than threshold
        if self.training and self.select_random and self.loss_ver[:3]=='ctc':
            rand_arr = peak_min + torch.rand_like(peak) * (threshold - peak_min)
            peak = torch.where(peak > threshold, peak, rand_arr)
        else:
            rand_arr = torch.arange(peak_min, threshold, peak.size(1), device=peak.get_device()).repeat(peak.size(0), 1)
            peak = torch.where(peak > threshold, peak, rand_arr)
            # peak = torch.where(peak > threshold, peak, torch.tensor(threshold, device=peak.get_device()))
        
        
        
        ##### 3.1. using only selected tokens (attention_mask ~= torch.ones)
        _, peak_indices = adaPool(peak)
        attention_mask = torch.ones_like(peak_indices, dtype=torch.long)
        word_embeddings = torch.gather(hidden_states_wav, 
                                       1, 
                                       peak_indices.unsqueeze(-1).expand(-1,-1,self.n_hidden))
        
        
        # ##### 3.2. using attention_mask
        # ##### NOTE: Not working when the wav2vec output is longer than 1024(=n_position)
        # peak_values, peak_indices = adaPool(peak)
        # attention_mask = torch.where(peak >= peak_values.min(dim=1)[0].unsqueeze(-1).expand(-1,peak.size(1)), 1, 0)
        # attention_mask = nn.functional.pad(attention_mask, (0,1,0,0), mode='constant', value=0)
        # word_embeddings = hidden_states_wav
        
        
        
        
        
        
        # lm_logits_0 = self.lm_head(word_embeddings)
        
        
        ############# Feature2Text : Referred from GPT2LMHeadModel #############

        transformer_outputs = self.transformer(
            input_ids=None, # Not used in this model
            past_key_values=None, # Not used in this model
            attention_mask=attention_mask,
            token_type_ids=None, # Not used in this model
            position_ids=None, # Not used in this model
            head_mask=None, # Not used in this model
            inputs_embeds=word_embeddings,
            encoder_hidden_states=None, # Not used in this model
            encoder_attention_mask=None, # Not used in this model
            use_cache=use_cache,
            output_attentions=output_attentions_gpt2,
            output_hidden_states=output_hidden_states_gpt2,
            return_dict=return_dict_gpt2,
        )
        
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism for GPT
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)


        ############# Computing Loss #############
        ############# CTC Loss : Referred from Wav2Vec2ForCTC #############

        loss = None
        if labels is not None:
            loss_ce, loss_ctc, loss_peak = 0.0, 0.0, 0.0
            
                        
            # # Flatten the tokens
            # loss_fct_0 = CrossEntropyLoss()
            # loss_ce = loss_fct_0(lm_logits_0.contiguous().view(-1, lm_logits_0.size(-1)), labels.contiguous().view(-1))
            
            
            
            # pred_label = torch.argmax(lm_logits_0, dim=2, keepdim=False)
            # loss_peak += torch.binary_cross_entropy_with_logits(torch.sigmoid(peak_values.contiguous()), 
            #                                                   (pred_label == labels).float())

            

#             if labels.max() >= self.config.vocab_size:
#                 raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

#             # log_probs = nn.functional.log_softmax(lm_logits_0, dim=-1, 
#             #                                         dtype=torch.float32).transpose(0, 1)  # ctc_loss doesn't support fp16
#             log_probs = nn.functional.log_softmax(lm_logits_0, dim=-1).transpose(0, 1)
        
#             # input_lengths for ctc_loss is defined from RNN peak detection
#             # this can be computed from attention_mask
#             input_lengths = (attention_mask > 0).sum(-1)
            
#             # assuming that padded tokens are filled with 'Ġ'
#             # unlike wav2vec we can get this information from given `output_attention_mask`
#             labels_mask = (output_attention_mask > 0)
#             target_lengths = labels_mask.sum(-1)
#             flattened_targets = labels.masked_select(labels_mask)
            
#             # See https://pytorch.org/docs/stable/generated/torch.nn.functional.ctc_loss.html
#             with torch.backends.cudnn.flags(deterministic = True):
#                 loss_ctc = nn.functional.ctc_loss(
#                     log_probs,
#                     flattened_targets,
#                     input_lengths,
#                     target_lengths,
#                     blank=self.config.pad_token_id,
#                     # reduction=self.config.ctc_loss_reduction,
#                     reduction='mean',
#                     zero_infinity=self.config.ctc_zero_infinity,
#                 )
                
    
    
            

        
            # DO NOT Shift
            shift_logits = lm_logits.contiguous()
            shift_labels = labels.contiguous()
            shift_labels = nn.functional.pad(shift_labels, [0,lm_logits.size(1)-labels.size(1),0,0], 
                                             mode='constant', value=self.config.pad_token_id)
            # # DO Shift
            # shift_logits = lm_logits[..., :-1, :].contiguous()
            # shift_labels = labels[..., 1:].contiguous()
            # shift_labels = nn.functional.pad(shift_labels, [0,lm_logits.size(1)-labels.size(1),0,0], 
            #                                  mode='constant', value=self.config.pad_token_id)
            
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss_ce = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
                
            # ctc_loss doesn't support fp16
            log_probs_1 = nn.functional.log_softmax(lm_logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            input_lengths_1 = (attention_mask > 0).sum(-1)
           
            labels_mask_1 = (output_attention_mask > 0)
            labels_mask_1[..., 0] = False
            target_lengths_1 = labels_mask_1.sum(-1)
            flattened_targets_1 = labels.masked_select(labels_mask_1)
            
            # See https://pytorch.org/docs/stable/generated/torch.nn.functional.ctc_loss.html
            with torch.backends.cudnn.flags(deterministic = True):
                loss_ctc = nn.functional.ctc_loss(
                    log_probs_1,
                    flattened_targets_1,
                    input_lengths_1,
                    target_lengths_1,
                    blank=self.config.pad_token_id,
                    # reduction=self.config.ctc_loss_reduction,
                    reduction='mean',
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                
            
            # loss = loss_ce + loss_ctc + loss_peak
            if self.training:
                if self.loss_ver == 'ctc':
                    loss = loss_ctc
                elif self.loss_ver == 'ce':
                    loss = loss_ce
                elif self.loss_ver == 'ctc-ce':
                    loss = loss_ctc
                    self.loss_ver = 'ce-ctc'
                elif self.loss_ver == 'ce-ctc':
                    loss = loss_ce
                    self.loss_ver = 'ctc-ce'
            else:
                loss = loss_ctc + loss_ce



        if not return_dict_gpt2:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
#             logits=[lm_logits_0, lm_logits],
            
#             past_key_values=transformer_outputs.past_key_values,
#             hidden_states=transformer_outputs.hidden_states,
#             attentions=transformer_outputs.attentions,
#             cross_attentions=transformer_outputs.cross_attentions,
        )





