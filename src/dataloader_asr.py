import torch
from dataset_asr import load_asr_dataset, DATASET_ARGS
from torch.utils.data import DataLoader

def preprocess_audio(audio_array, sample_rate, feature_extractor, max_audio_length=None):
    if max_audio_length is None:
        max_audio_length = audio_array.shape[0]
    if feature_extractor.__class__.__name__ in ['Wav2Vec2FeatureExtractor']:
        return feature_extractor(audio_array, 
                                 sampling_rate=sample_rate, 
                                 return_tensors='pt',
                                 padding='max_length',
                                 max_length=max_audio_length).input_values
    elif feature_extractor.__class__.__name__ in ['WhisperFeatureExtractor']:
        return feature_extractor(audio_array, 
                                 sampling_rate=sample_rate, 
                                 return_tensors='pt',
                                 padding='max_length',
                                 max_length=max_audio_length).input_features


def preprocess_text(text, tokenizer_asr, tokenizer_llm):
    return tokenizer_asr(text, return_tensors='pt'), \
           tokenizer_llm(text, return_tensors='pt')


def collate_fn(batch, feature_extractor, tokenizer_asr, tokenizer_llm,
               audio_column=0, text_column=1):
    
    audio = [preprocess_audio(item[audio_column]['array'], 
                              item[audio_column]['sampling_rate'], 
                              feature_extractor) for item in batch]
    texts = [item[text_column] for item in batch]
    text_asr, text_llm = zip(*[preprocess_text(text,
                                               tokenizer_asr, tokenizer_llm) for text in texts])

    max_audio_len = max([item.shape[-1] for item in audio])
    
    audio_attention_mask = torch.stack([torch.nn.functional.pad(torch.ones(item.shape[-1], dtype=torch.int),
                                                                (0, max_audio_len - item.shape[-1]),
                                                                value=0) for item in audio])
    
    audio = torch.concat([torch.nn.functional.pad(item, 
                                                  (0, max_audio_len - item.shape[-1]),
                                                  value=feature_extractor.padding_value) for item in audio], dim=0)
    
    max_token_len_asr = max([item['input_ids'].shape[1] for item in text_asr])
    token_asr = torch.concat([torch.nn.functional.pad(item.input_ids,
                                                     (0, max_token_len_asr - item.input_ids.shape[1]),
                                                     value=tokenizer_asr.pad_token_id) for item in text_asr], dim=0)
    token_asr_attention_mask = torch.concat([torch.nn.functional.pad(item.attention_mask,
                                                                    (0, max_token_len_asr - item.attention_mask.shape[1]),
                                                                    value=0) for item in text_asr], dim=0)
    
    max_token_len_llm = max([item['input_ids'].shape[1] for item in text_llm])
    token_llm = torch.concat([torch.nn.functional.pad(item.input_ids,
                                                     (0, max_token_len_llm - item.input_ids.shape[1]),
                                                     value=tokenizer_llm.pad_token_id) for item in text_llm], dim=0)
    token_llm_attention_mask = torch.concat([torch.nn.functional.pad(item.attention_mask,
                                                                    (0, max_token_len_llm - item.attention_mask.shape[1]),
                                                                    value=0) for item in text_llm], dim=0)
    
    return {'audio': audio, 'audio_attention_mask': audio_attention_mask, 
            'labels': {
                'original_text': texts,
                'token_ids_asr': token_asr, 'attn_mask_asr': token_asr_attention_mask,
                'token_ids_llm': token_llm, 'attn_mask_llm': token_llm_attention_mask
            }}



def collate_fn_asr2llm(asr_model_name, llm_model_name, cache_dir=None, token=None):
    # import os
    # print('Current user:')
    # os.system('huggingface-cli whoami')
    if 'facebook/wav2vec2' in asr_model_name:
        from transformers import Wav2Vec2Processor
        feature_extractor_asr = Wav2Vec2Processor.from_pretrained(asr_model_name,
                                                                  cache_dir=cache_dir).feature_extractor
        tokenizer_asr = Wav2Vec2Processor.from_pretrained(asr_model_name,
                                                          cache_dir=cache_dir).tokenizer
    elif 'openai/whisper' in asr_model_name:
        from transformers import WhisperProcessor
        feature_extractor_asr = WhisperProcessor.from_pretrained(asr_model_name,
                                                                 cache_dir=cache_dir).feature_extractor
        tokenizer_asr = WhisperProcessor.from_pretrained(asr_model_name,
                                                         cache_dir=cache_dir).tokenizer
    else:
        assert False, f'Unsupported ASR model: {asr_model_name}'

    if 'mistralai/' in llm_model_name:
        from transformers import AutoProcessor
        tokenizer_llm = AutoProcessor.from_pretrained(llm_model_name,
                                                      cache_dir=cache_dir,
                                                      token=token)
    elif 'openai-community/gpt2' in llm_model_name:
        from transformers import GPT2TokenizerFast
        tokenizer_llm = GPT2TokenizerFast.from_pretrained(llm_model_name,
                                                          cache_dir=cache_dir)
    else:
        assert False, f'Unsupported LLM model: {llm_model_name}'

    return lambda batch: collate_fn(batch, feature_extractor_asr, tokenizer_asr, tokenizer_llm)


def load_dataloader(dataset_name, split, asr_model_name, llm_model_name, 
                    cache_dir=None, token=None, batch_size=1):
    phase_name = DATASET_ARGS[dataset_name]['phase'][split]
    shuffle = (split == 'train')
    dataset = load_asr_dataset(dataset_name, phase_name, cache_dir=cache_dir)
    collate_fn = collate_fn_asr2llm(asr_model_name, llm_model_name, cache_dir, token)

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)



if __name__ == '__main__':
    from utils import set_huggingface_cache_dir
    from pprint import pprint

    cache_dir = "/data/yoom618/datasets/"
    dataset_name = "ami"
    batch_size = 3
    
    asr_model_name = "facebook/wav2vec2-base-960h"  # (batch_size, audio_length)
    # asr_model_name = "openai/whisper-small"       # (batch_size, 80, audio_length)

    # llm_model_name = "openai-community/gpt2"
    llm_model_name = "mistralai/Mistral-Nemo-Instruct-2407"

    token = set_huggingface_cache_dir("/data/yoom618/datasets/")


    test_loader = load_dataloader(dataset_name, 
                                  'test', 
                                  asr_model_name, 
                                  llm_model_name, 
                                  cache_dir, 
                                  token, 
                                  batch_size)

    for batch in test_loader:
        pprint(batch)
        print(f'audio: {batch["audio"].shape}')
        print(f'audio_attention_mask: {batch["audio_attention_mask"].shape}')
        print(f'token_asr: {batch["token_asr"].shape}')
        print(f'token_asr_attention_mask: {batch["token_asr_attention_mask"].shape}')
        break
