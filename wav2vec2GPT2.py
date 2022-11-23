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
        
        self.n_hidden = config.hidden_size
        self.vocab_size = config.vocab_size
        
        self.wav2vec2 = Wav2Vec2Model2(config)
        self.dropout = nn.Dropout(config.final_dropout)
        
        self.rnn_compressor = nn.GRU(input_size=config.hidden_size, 
                                     hidden_size=config.hidden_size, 
                                     # hidden_size=(config.hidden_size + 1),  
                                     num_layers=1,
                                     dropout=0.1,
                                     batch_first=True)
        self.adaPool = nn.AdaptiveMaxPool1d(config.n_positions, return_indices=True)
        
        ### 1.
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.model_parallel = False
        self.device_map = None
        ### 2.
        # self.wte = nn.Embedding(config.vocab_size, config.n_embd)

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
        for param in self.transformer.parameters():
            param.requires_grad = False
    
    def freeze_lm_head(self):
        for param in self.lm_head.parameters():
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
        for param in self.transformer.parameters():
            param.requires_grad = True
    
    def unfreeze_lm_head(self):
        for param in self.lm_head.parameters():
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
            output_attentions=output_attentions_wav2vec,
            output_hidden_states=output_hidden_states_wav2vec,
            return_dict=return_dict_wav2vec,
        )

        hidden_states = outputs.last_hidden_state
        hidden_states = self.dropout(hidden_states) # size: (batch_size, seq_len_from_adapter, config.hidden_states=768)
        
        
        ############# Feature Propagation via CNN+RNN : Proposed #############

        hidden_states, _ = self.rnn_compressor(hidden_states) # size: (batch_size, seq_len, config.hidden_states)
        
        
        ############# RNNFeature2TextEmbedding & AdaptivePooling : Proposed ############
        
        threshold = 0.8
        lm_logits_all = self.lm_head(hidden_states)  # torch.matmul(hidden_states, self.transformer.wte.weight.transpose(1,0))
        sim, _ = nn.functional.softmax(lm_logits_all, dim=-1).max(dim=-1)
        sim = nn.functional.threshold(sim, threshold, 0, inplace=False) # threshold 안 넘으면 같이 샘플링 됨
        _, peak_indices = self.adaPool(sim) # 가장 사전에 있을 법한 단어만 선정
        
        lm_logits = torch.gather(lm_logits_all, 
                                 1, 
                                 peak_indices.unsqueeze(-1).expand(-1,-1,self.vocab_size))
#         word_embeddings = torch.gather(hidden_states, 
#                                        1, 
#                                        peak_indices.unsqueeze(-1).expand(-1,-1,self.n_hidden))
        
#         ############ Feature2Text : Referred from GPT2LMHeadModel #############

#         transformer_outputs = self.transformer(
#             input_ids=None, # Not used in this model
#             past_key_values=None, # Not used in this model
#             attention_mask=attention_mask,
#             token_type_ids=None, # Not used in this model
#             position_ids=None, # Not used in this model
#             head_mask=None, # Not used in this model
#             inputs_embeds=word_embeddings,
#             encoder_hidden_states=None, # Not used in this model
#             encoder_attention_mask=None, # Not used in this model
#             use_cache=use_cache,
#             output_attentions=output_attentions_gpt2,
#             output_hidden_states=output_hidden_states_gpt2,
#             return_dict=return_dict_gpt2,
#         )



        
        ############# Computing Loss #############
        
        loss = None
        if labels is not None:
            loss_ce, loss_ctc, loss_peak = 0.0, 0.0, 0.0
            
            
            ############# CE Loss : Referred from GPT2LM #############
            
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss_ce = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            
            
            
            
            
            
            
            loss = loss_ce + loss_ctc + loss_peak
            
            
        ######################################################################    

        if not return_dict_gpt2:
            output = (lm_logits_all,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, 
            logits=lm_logits_all, 
            peak_indices=peak_indices, 
            # attention_mask=attention_mask,
            hidden_states=outputs.hidden_states, 
        )