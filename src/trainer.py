


from torch import nn
from transformers import Trainer
from loss import compute_ctc_loss

class CTCTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    def compute_loss(self, model, inputs, return_outputs=False):
        label = inputs.pop('labels')
        # print("Label: ", label)
        # print(label['original_text'], label['token_ids_llm'].shape)
        model_output = model(**inputs)
        # print("Model Output: ", model_output)
        loss = compute_ctc_loss(self.criterion, model_output, label)
        # print("Loss: ", loss)

        return (loss, model_output) if return_outputs else loss




if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    from utils import set_huggingface_cache_dir
    from dataset_asr import load_asr_dataset, DATASET_ARGS
    from dataloader_asr import collate_fn_asr2llm
    # from model import Wav2Vec2Mistral
    from model_splited import *


    from transformers import AutoModel
    import transformers
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    ########## HYPERPARAMETERS ##########
    cache_dir = "/data/yoom618/datasets/"
    dataset_name = "ami"
    batch_size = 6

    # asr_model_name = "facebook/wav2vec2-base-960h"
    asr_model_name = "facebook/wav2vec2-base"
    # asr_model_name = "openai/whisper-small"

    # llm_model_name = "openai-community/gpt2"
    # llm_model_name = "mistralai/Mistral-Nemo-Instruct-2407"
    llm_model_name = "mistralai/Mistral-7B-v0.1"

    #####################################


    # Set huggingface cache directory
    token = set_huggingface_cache_dir(cache_dir)

    # Load data
    train_dataset = load_asr_dataset(
        name=dataset_name,
        phase = DATASET_ARGS[dataset_name]['phase']['train'],
        cache_dir=cache_dir,
        token=token
    )

    valid_dataset = load_asr_dataset(
        name=dataset_name,
        phase = DATASET_ARGS[dataset_name]['phase']['valid'],
        cache_dir=cache_dir,
        token=token
    )

    collate_fn = collate_fn_asr2llm(
        asr_model_name=asr_model_name,
        llm_model_name=llm_model_name,
        cache_dir=cache_dir,
        token=token
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )
    
    model_asr = AutoModel.from_pretrained(asr_model_name,
                                          cache_dir=cache_dir,
                                          token=token)
    model_llm = AutoModel.from_pretrained(llm_model_name,
                                          cache_dir=cache_dir,
                                          token=token)

    # model = Wav2Vec2Mistral(model_asr, model_llm.embed_tokens, model_llm.rotary_emb, llm_input_dim=4096)
    model = Wav2Vec2Mistral(model_asr, model_llm)


    step_length = (len(train_dataset) - 1) // batch_size + 1
    training_args = transformers.TrainingArguments(
        # output_dir=os.path.join(cache_dir, "testing/output"),
        # logging_dir=os.path.join(cache_dir, "testing/logs"),
        output_dir="/home/yoom618/ASR-w-GPT/scratch/output",
        logging_dir="/home/yoom618/ASR-w-GPT/scratch/logs",

        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,

        num_train_epochs=1,
        learning_rate=1e-4,
        warmup_steps=step_length * 2,
        weight_decay=0.001,
        lr_scheduler_type="cosine",

        eval_strategy='epoch',
        eval_steps=1,
        
        logging_strategy='steps',
        logging_steps= step_length // 10,

        save_strategy='epoch',
        save_total_limit=3,
        overwrite_output_dir=True,
    )

    trainer = CTCTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_fn,
    )

    # print(trainer.args)

    trainer.train()


        
