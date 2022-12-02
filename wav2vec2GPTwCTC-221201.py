import torch
import torch.nn as nn
import torch.nn.functional as F

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
        ### self.feature_projection = Wav2Vec2FeatureProjection(config)  # REMOVED

        self.adapter = Wav2Vec2Adapter(config) if config.add_adapter else None

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
        attention_mask=None, # not used in this model since no Encoder is given
        mask_time_indices=None, # not used in this model since no Encoder is given
        output_attentions=None, # not used in this model since no Encoder is given
        output_hidden_states=None,
        return_dict=None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        ### hidden_states, _ = self.feature_projection(extract_features)  # REMOVED

        if self.adapter is not None:
            # hidden_states = self.adapter(hidden_states)  # REMOVED
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
                # last_hidden_state=hidden_states,  # REMOVED
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
        
        
        if isinstance(config.output_hidden_size, int):
            config.output_hidden_size = [config.output_hidden_size] * config.num_adapter_layers
        
        self.wav2vec2 = Wav2Vec2Model2(config)
        self.dropout = nn.Dropout(config.final_dropout)
        
        compress_input_size = config.output_hidden_size[config.num_adapter_layers - 1]
        
        ##### 1. CNN Module
        self.compressor = nn.Conv1d(
            in_channels=compress_input_size, 
            out_channels=config.hidden_size, 
            kernel_size=3, stride=1, bias=True,
        )
        
        ##### 2. ATTN Module
        self.attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size, 
                nhead=8,
                dim_feedforward=4 * config.hidden_size,
                batch_first=True
            ), num_layers=1)


        self.n_hidden = config.hidden_size
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.ctc_loss_reduction = config.ctc_loss_reduction
        self.ctc_zero_infinity = config.ctc_zero_infinity
        self.select_random = config.select_random
        self.loss_ver = config.loss_ver
        

        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)


        ##### Initialize weights and apply final processing
        self.post_init()
        

    
    
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

        hidden_states_wav = self.wav2vec2(
            input_values,
            attention_mask=input_attention_mask,
            output_attentions=output_attentions_wav2vec,
            output_hidden_states=output_hidden_states_wav2vec,
            return_dict=return_dict_wav2vec,
        ).last_hidden_state

#         hidden_states_wav = self.dropout(hidden_states_wav)
        
        ############# Feature Propagation : Proposed #############
        
        ##### 1. CNN Module
        hidden_states_wav = self.compressor(hidden_states_wav.transpose(1,2)).transpose(1,2)
        hidden_states_wav = self.dropout(hidden_states_wav)
#         # ##### 2. ATTN Module
#         hidden_states_wav = self.attn(hidden_states_wav)
        

        ############# Peak Detection : Proposed #############
        
        ##### 1.1. cos difference
        # NOTE: the cosine values are almost positive (IDK why)
        # https://vaibhavgarg1982.medium.com/why-are-cosine-similarities-of-text-embeddings-almost-always-positive-6bd31eaee4d5
        peak_min, threshold, peak_max = 1.-(1.), 1.-(.5), 1.-(-.5)
        cos = nn.CosineSimilarity(dim=-1, eps=1e-10)
        peak = 1 - cos(hidden_states_wav[..., :-1, :], hidden_states_wav[..., 1:, :])
        
#         ##### 1.2. label difference
#         # NOTE: Skip pad token
#         # NOTE: Not using log_softmax since we have to compare the exact probability values
#         peak_min, threshold, peak_max = -1.0, 0.5, 1.0
#         peak = nn.functional.softmax(self.lm_head(hidden_states_wav), dim=-1)
#         peak = peak[..., :, np.arange(self.vocab_size) != self.pad_token_id]
#         peak = (peak[..., :-1, :] - peak[..., 1:, :]).max(dim=-1)[0]
        
        # ##### 2.1. using all N tokens
        # pass
        
        ##### 2.2. use only upper than threshold
        # NOTE: torch.linspace()'s default requires_grad is `False`
        batch_size, seq_len = peak.size()
        
        if not (self.training and self.select_random):  ### pick back sequentially
            rand_arr = torch.linspace(peak_min, threshold, seq_len, device=peak.get_device()).repeat(batch_size, 1)
        elif self.loss_ver[:3]=='ctc':
#             ### RANDOM
#             rand_arr = peak_min + torch.rand_like(peak) * (threshold - peak_min)
            # ### pick back sequentially
            # rand_arr = torch.linspace(peak_min, threshold, seq_len, device=peak.get_device()).repeat(batch_size, 1)
            ### pick front sequentially
            rand_arr = torch.linspace(threshold, peak_min, seq_len, device=peak.get_device()).repeat(batch_size, 1)
        else:  # 'ce'
            # ### RANDOM
            # rand_arr = peak_min + torch.rand_like(peak) * (threshold - peak_min)
            ### pick back sequentially
            rand_arr = torch.linspace(peak_min, threshold, seq_len, device=peak.get_device()).repeat(batch_size, 1)
            # ### pick front sequentially
            # rand_arr = torch.linspace(threshold, peak_min, seq_len, device=peak.get_device()).repeat(batch_size, 1)
        peak = torch.where(peak > threshold, peak, rand_arr)
        
        
        
        ########## 3. select maximum token
        topk_size = 48
        
        _, peak_indices = peak.topk(topk_size, dim=-1, largest=True, sorted=False)
        attention_mask = torch.ones_like(peak_indices, dtype=torch.long)
        word_embeddings = torch.gather(hidden_states_wav, 
                                       1, 
                                       peak_indices.unsqueeze(-1).expand(-1,-1,self.n_hidden))
        
        
        word_embeddings = self.attn(word_embeddings)
        lm_logits_0 = self.lm_head(word_embeddings)
        # attention_mask = (lm_logits_0.argmax(dim=-1) != self.pad_token_id).int()   # NOT WORKING
        
        
        ############# Feature2Text : Referred from GPT2LMHeadModel #############
        
        word_embeddings_gathered = list()
        pred_ids_0 = lm_logits_0.argmax(dim=-1)
        for i in range(batch_size):
            j = torch.where(pred_ids_0[i] != self.pad_token_id)[0]
            word_embeddings_gathered.append(F.pad(word_embeddings[i,j,:], [0,0,0,topk_size-len(j)]))
        word_embeddings_gathered = torch.stack(word_embeddings_gathered)
        
        transformer_outputs = self.transformer(
            input_ids=None, # Not used in this model
            past_key_values=None, # Not used in this model
            attention_mask=None,
            # attention_mask=attention_mask,
            # attention_mask=output_attention_mask,
            token_type_ids=None, # Not used in this model
            position_ids=None, # Not used in this model
            head_mask=None, # Not used in this model
#             inputs_embeds=word_embeddings,
            inputs_embeds=word_embeddings_gathered,
            encoder_hidden_states=None, # Not used in this model
            encoder_attention_mask=None, # Not used in this model
            use_cache=use_cache,
            output_attentions=output_attentions_gpt2,
            output_hidden_states=output_hidden_states_gpt2,
            return_dict=return_dict_gpt2,
        )

        lm_logits_1 = self.lm_head(transformer_outputs[0])


        ############# Computing Loss #############
        ############# CTC Loss : Referred from Wav2Vec2ForCTC #############
        
        if labels is None:
            return CausalLMOutputWithCrossAttentions(
                loss=None,
                logits=[lm_logits_0, lm_logits_1],
            )

        if (not self.training) or (self.loss_ver[:3]=='ctc'):
            if labels.max() >= self.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.vocab_size}")

            # ctc_loss doesn't support fp16
            log_probs = F.log_softmax(lm_logits_0, dim=-1, dtype=torch.float32).transpose(0, 1)
        
            # input_lengths for ctc_loss is defined from peak detection
            # this can be computed from attention_mask
            input_lengths = (attention_mask > 0).sum(-1)
            
            # assuming that padded tokens are filled with `pad_token`
            # unlike wav2vec we can get this information from given `output_attention_mask`
            labels_mask = (output_attention_mask > 0)
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)
            
            # See https://pytorch.org/docs/stable/generated/torch.nn.functional.ctc_loss.html
            with torch.backends.cudnn.flags(deterministic = True):
                loss_ctc = F.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.pad_token_id,
                    reduction=self.ctc_loss_reduction,
                    zero_infinity=self.ctc_zero_infinity,
                )
                

        if (not self.training) or (self.loss_ver[:2]=='ce'):
            
#             ### 1. DO NOT Shift
#             loss_fct = CrossEntropyLoss()
#             loss_ce = loss_fct(lm_logits_1.contiguous().view(-1, lm_logits_1.size(-1)), 
#                                F.pad(labels.contiguous(), (0,lm_logits_1.size(1)-labels.size(1),0,0)).view(-1))      
            
            ### 2. DO Shift
            shift_logits = lm_logits_1[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss_ce = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                               F.pad(shift_labels, (0,shift_logits.size(1)-shift_labels.size(1),0,0)).view(-1))            
                
#             # ctc_loss doesn't support fp16
#             output_attention_mask = output_attention_mask[..., 1:]
#             attention_mask = attention_mask[..., :-1]
#             log_probs_1 = F.log_softmax(shift_logits, dim=-1, dtype=torch.float32).transpose(0, 1)
#             seq_lengths_1 = (attention_mask > 0).sum(-1)
           
#             labels_mask_1 = (output_attention_mask > 0)
#             target_lengths_1 = labels_mask_1.sum(-1)
#             flattened_targets_1 = shift_labels.masked_select(labels_mask_1)
            
#             # See https://pytorch.org/docs/stable/generated/torch.nn.functional.ctc_loss.html
#             with torch.backends.cudnn.flags(deterministic = True):
#                 loss_ce = nn.functional.ctc_loss(
#                     log_probs_1,
#                     flattened_targets_1,
#                     seq_lengths_1,
#                     target_lengths_1,
#                     blank=self.pad_token_id,
#                     reduction=self.ctc_loss_reduction,
#                     zero_infinity=self.ctc_zero_infinity,
#                 )


        # loss = loss_ce + loss_ctc
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
            output = (lm_logits_1,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=[lm_logits_0, lm_logits_1],
            ### past_key_values=transformer_outputs.past_key_values,  # REMOVED
            ### hidden_states=transformer_outputs.hidden_states,  # REMOVED
            ### attentions=transformer_outputs.attentions,  # REMOVED
            ### cross_attentions=transformer_outputs.cross_attentions,  # REMOVED
        )





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
    
    def freeze_compressor(self):
        self.compressor.eval()
        for param in self.compressor.parameters():
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
    
    def unfreeze_compressor(self):
        self.compressor.train()
        for param in self.compressor.parameters():
            param.requires_grad = True
    
    def unfreeze_gpt_decoder(self):
        self.transformer.train()
        for param in self.transformer.parameters():
            param.requires_grad = True
    
    def unfreeze_lm_head(self):
        self.lm_head.train()
        for param in self.lm_head.parameters():
            param.requires_grad = True
    


