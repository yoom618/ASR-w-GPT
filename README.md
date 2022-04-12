# Wav-BERT

based on fairseq framework(https://github.com/pytorch/fairseq)  
# code structure

```
--fairseq
    --bert                                                  :BERT related codes
        --modeling.py                                       :BERT pytorch implementation
        --tokenization.py                                   :BERT tokenizer
        --file_utils.py                                     :BERT read ckpts etc utils 
    --examples/speech_recognition                           :inference related codes
        --infer_cif_bert.py                                 :wav-cif-bert infer entrance
        --cif_bert_decoder.py                               :wav-cif-bert infer implementation
        --infer_wav2bert_different_token_v2_ce.py           :Linguistic-Guide infer entrance
        --wav2bert_decoder_different_token_v2_ce.py         :Linguistic-Guide infer implementation
        --infer_wav2bert_different_token_v2.py              :Acoustic-Guide infer entrance
        --wav2bert_decoder_different_token_v2.py            :Acoustic-Guide infer implementation
        --infer_wav2bert_different_token_v2_two_way.py      :Cross-Modal infer entrance
        --wav2bert_decoder_different_token_v2_two_way.py    :Cross-Modal infer implementation
        --infer_wav2bert_different_token_v2_v1_fusion.py    :Adapter-BERT infer entrance
        --wav2bert_decoder_different_token_v2_fusion_v1.py  :Adapter-BERT infer implementation
    --fairseq
        --criterions                                          :criterion related codes
            --ctc2_mlm.py                                     :Acoustic-Guide criterion
            --ctc_ce_mlm.py                                   :Linguistic-Guide criterion
            --ctc_two_way.py                                  :Cross-Modal criterion
            --ctc2_mlm_fusion_v1.py                           :Adapter-BERT criterion
            --qua_ce_acc.py                                   :wav-cif-bert criterion
            --ctc_two_way_pe.py                               :Embedding Replacement criterion
        --data                                                :dataset related codes
            --wav2bert_dataset.py                             :wav-bert dataset
            --cif_add_target_dataset.py                       :wav-cif-bert dataset
        --task                                                  :training task related codes
            --audio_cif_bert.py                                 :wav-cif-bert task
            --wav2bert_task.py                                  :wav-bert task
        --models/wav2vec                                        :network implementation codes
            --w2v_cif_bert.py                                   :wav-cif-bert model code
            --wav2bert_mask_predict_fusion_two_way_gate_ctc_to_bert.py  ：Cross-Modal model code
            --wav2bert_mask_predict_fusion_two_way_ctc_to_bert.py       :Cross-Modal without gate model code
            --wav2bert_mask_predict_fusion_two_way_gate_ctc_to_bert_pe.py   :Embedding Replacement model code
            --wav2bert_masked_predict_fusion_ce_gate_ctc_to_bert.py         :Linguistic-Guide model code
            --wav2bert_masked_predict_fusion_ctc_gate_ctc_to_bert.py        :Acoustic-Guide model code
            --wav2bert_mask_predict_fusion_ctc_gate_ctc_to_bert_fusion_v1.py    ：Adapter-BERT model code
```




# Training


eg:   
**training swahili cross-model**
```
exp_name=swahili_cross_model_1 && \
dataset_dir=/code/manifest/202-swahili/ && \
valid_subset=eval && \
load_ckpt=wav2vec_small.pt && bert_model=bert_multi \
model_name=wav2bert_masked_predict_fusion_two_way_gate_ctc_to_bert && \
output_path=./output && \
python train.py --distributed-world-size 1 $dataset_dir --save-dir $output_path/ckpts/$exp_name --fp16 \
--post-process bert_bpe_piece --valid-subset $valid_subset --no-epoch-checkpoints --best-checkpoint-metric wer --num-workers 4 \
--max-update 200000 --sentence-avg --task wav2bert_task --arch $model_name --sample-rate 16000 \
--w2v-path $load_ckpt --update-freq 4 --ctc-weight 0.5 --mlm-weight 0.5 --fusion-v2 \
--labels wrd --apply-mask --mask-selection static --mask-other 0 --mask-length 10 --mask-prob 0.65 --layerdrop 0.1 \
--mask-channel-selection static --mask-channel-other 0 --mask-channel-length 64 --mask-channel-prob 0.5 --zero-infinity \
--feature-grad-mult 1.0 --freeze-finetune-updates 10000 --validate-after-updates 10000 --optimizer adam \
--adam-betas '(0.9, 0.98)' --adam-eps 1e-08 --lr 5e-05 --lr-scheduler tri_stage --warmup-steps 10000 --hold-steps 90000 \
--decay-steps 100000 --final-lr-scale 0.05 --final-dropout 0.0 --dropout 0.0 --activation-dropout 0.1 --criterion ctc_two_way \
--attention-dropout 0.0 --max-tokens 640000 --seed 2337 --log-format json --log-interval 500 --ddp-backend no_c10d \
--adapter-dimension 512 --decoder-bert-model-name $bert_model --tensorboard-logdir $output_path/tensorboard/$exp_name \
--different-tokens-v2 --lexicon $bert_model/dict.multi.txt --add-input \
--mix-ctc-deocde-prob-range "(0.1,0.9)" --mix-ctc-step-range "(100000, 200000)" --fuse-input --fuse-input-gate \
--validate-interval 5 \
```
**training aishell cross-model**
```
exp_name=aishell_cross_model_1 && \
dataset_dir=/code/manifest/aishell && \
valid_subset=dev && load_ckpt=wav2vec_small.pt && bert_model=bert_chinese \
model_name=wav2bert_masked_predict_fusion_two_way_gate_ctc_to_bert && \
output_path=./output && \
python train.py --distributed-world-size 1 $dataset_dir --save-dir $output_path/ckpts/$exp_name --fp16 \
--post-process letter --valid-subset $valid_subset --no-epoch-checkpoints --best-checkpoint-metric wer --num-workers 4 \
--max-update 200000 --sentence-avg --task wav2bert_task --arch $model_name --sample-rate 16000 \
--w2v-path $load_ckpt --update-freq 4 --ctc-weight 0.5 --mlm-weight 0.5 --fusion-v2 \
--labels ltr --apply-mask --mask-selection static --mask-other 0 --mask-length 10 --mask-prob 0.65 --layerdrop 0.1 \
--mask-channel-selection static --mask-channel-other 0 --mask-channel-length 64 --mask-channel-prob 0.5 --zero-infinity \
--feature-grad-mult 1.0 --freeze-finetune-updates 10000 --validate-after-updates 10000 --optimizer adam \
--adam-betas '(0.9, 0.98)' --adam-eps 1e-08 --lr 5e-05 --lr-scheduler tri_stage --warmup-steps 10000 --hold-steps 90000 \
--decay-steps 100000 --final-lr-scale 0.05 --final-dropout 0.0 --dropout 0.0 --activation-dropout 0.1 --criterion ctc_two_way \
--attention-dropout 0.0 --max-tokens 640000 --seed 2337 --log-format json --log-interval 500 --ddp-backend no_c10d \
--adapter-dimension 512 --decoder-bert-model-name $bert_model --tensorboard-logdir $output_path/tensorboard/$exp_name \
--lexicon $bert_model/dict.en.txt --add-input \
--mix-ctc-deocde-prob-range "(0.1,0.9)" --mix-ctc-step-range "(100000, 200000)" --fuse-input --fuse-input-gate --period-index 119 --chinese-cer
```

**training wav-cif-bert**
```
exp_name="swahili_cif_1"
W2V_PATH=wav2vec_small.pt
BERT="bert_multi"
SAVE_DIR=cif_ckpts/$exp_name
TENSORBOARD_DIR=cif_events/$exp_name
DATA_DIR=manifest/202-swahili
label_type=wrd

python train.py $DATA_DIR \
--save-dir $SAVE_DIR --tensorboard-logdir $TENSORBOARD_DIR \
--train-subset train --valid-subset valid --no-epoch-checkpoints \
--labels $label_type --num-workers 4 --max-update 200000 \
--lambda-ctc 2.0 --lambda-qua 0.2 --lambda-am 0.8 --lambda-lm 0.2 \
--arch w2v_cif_bert --task audio_cif_bert --criterion nar_qua_ctc_ce --best-checkpoint-metric uer \
--w2v-path $W2V_PATH --bert-name $BERT --infer-threash 0.8 \
--apply-mask --mask-selection static --mask-other 0 --mask-length 2 --mask-prob 0.1 --layerdrop 0.1 \
--mask-channel-selection static --mask-channel-other 0 --mask-channel-length 64 --mask-channel-prob 0.5 \
--feature-grad-mult 0.0 --freeze-finetune-updates 10000 --freeze-lm-finetune-updates 10000 \
--gold-rate-steps '(10000, 80000)' --gold-rate-range '(0.9, 0.2)' \
--validate-after-updates 10000  --validate-interval 4 --save-interval 4 \
--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-08 --lr 5e-05 --lr-scheduler tri_stage \
--warmup-steps 10000 --hold-steps 90000 --decay-steps 100000 --final-lr-scale 0.05 \
--final-dropout 0.0 --dropout 0.0 --activation-dropout 0.1 \
--attention-dropout 0.0 --max-tokens 640000 --seed 2337 --ddp-backend no_c10d --update-freq 4 \
--log-format json --log-interval 500
```

**main  hyper param**  
* distributed-world-size：gpu num used for training
* dataset_dir： manifest meta file path
* save-dir： ckpt output path
* tensorboard-logdir： tensorboard output path
* fp16： fp16 precision training
* post-process： post process for test，babel using bert_bpe_piece， aishell using letter
* labels: lable type for training text
* valid-subset： name of validset split
* best-checkpoint-metric： metric for saving best ckpt
* max-update： max training step
* task： training task
* arch: model implementation
* w2v-path： pretrain model path
* update-freq： accumulate gradient times
* ctc-weight， mlm-weight etc ： criterion loss weight, default 0.5
* feature-grad-mult： gradient multi coefficient for pre conv layer
* freeze-finetune-updates： freeze wav2vec feature extractor steps
* criterion： training criterion
* warmup-steps hold-steps decay-steps: lr policy steps
* lr： peak learning rate
* max-tokens： dynamic batch size(max sample point for audio in one batch)
* decoder-bert-model-name： bert dir path
* different-tokens-v2： token type for vocab
* lexicon： lexicon file
* add-input：adding eos and bos to text
* mix-ctc-deocde-prob-range， mix-ctc-step-range：using Sampling With Decay
* fuse-input， fuse-input-gate：using Embedding Attention Module
* chinese-cer：calculate cer


# Inference
eg:  
**infer aishell**
```
exp_name=aishell_cross_model_1 && \
python examples/speech_recognition/infer_wav2bert_different_token_v2_two_way.py manifest/aishell --task wav2bert_task \
--path ckpts/$exp_name/checkpoint_best.pt \
--gen-subset test \
--results-path outputs/$exp_name \
--criterion ctc_ce_mlm --max-tokens 2000000 \
--lexicon /wav2bert/bert_chinese/dict.en.txt --post-process letter --labels ltr \
--decoder-bert-model-name ./bert_chinese --left-pad-source False \
--test
```

**infer swahili**
```
python examples/speech_recognition/infer_wav2bert_different_token_v2_two_way.py ./manifest/202-swahili --task wav2bert_task \
--path ckpts/$exp_name/checkpoint_best.pt \
--gen-subset eval \
--results-path outputs/$exp_name \
--criterion ctc_ce_mlm --max-tokens 2000000 \
--lexicon bert_multi/dict.multi.txt --post-process letter --labels ltr \
--decoder-bert-model-name ./bert_multi --left-pad-source False \
--test --different-tokens-v2 
```

**main hyper param**
* path：ckpt path
* gen-subset：test split name
* lexicon： lexicon file
* different-tokens-v2： token type flag for babel
* post-process： post process for text
* labels：lable tyep
* decoder-bert-model-name： bert model path

**infer wav-cif-bert**
```
label_type=wrd
DATA_DIR=manifest/202-swahili 
BERT='bert_multi'
data_name=valid
MODEL_PATH=swahilli_cif_1/checkpoint_best.pt
RESULT_DIR=cif_results/swahilli_cif_1/

python ./examples/speech_recognition/infer_cif_bert.py $DATA_DIR \
--task audio_cif_bert --path $MODEL_PATH --bert-name $BERT \
--gen-subset $data_name --results-path $RESULT_DIR \
--criterion nar_qua_ctc_ce --labels $label_type --max-tokens 160000 --infer-threshold 0.8

```


# some other baseline

for training or infering the acoustic-guided, linguistic-guided, or Adapter-BERT, only need to replace the arch and criterion param in above script with specific method options, which is mentioned in the code structure part.


### Finetune BERT and use for BERT Rescore

**finetune**
* arch:wav2bert_onlybert
* criterion:bert_mlm
```
dataset_dir=manifest/202-swahili/ && \
exp_name=finetune_mbert_swahili_1 && valid_subset=valid && load_ckpt=wav2vec_small.pt && \
model_name=wav2bert_onlybert && bert_model=./bert_multi && output_path=./output  && \
python train.py --distributed-world-size 1 $dataset_dir --save-dir $output_path/ckpts/$exp_name --fp16 \
--post-process bert_bpe_piece --valid-subset $valid_subset --no-epoch-checkpoints --best-checkpoint-metric wer --num-workers 4 \
--max-update 200000 --sentence-avg --task wav2bert_task --arch $model_name --sample-rate 16000 \
--w2v-path $load_ckpt --update-freq 4 --ctc-weight 0 --encoder-ctc-weight 0 --mlm-weight 1 --fusion-v2 \
--labels wrd --apply-mask --mask-selection static --mask-other 0 --mask-length 10 --mask-prob 0.65 --layerdrop 0.1 \
--mask-channel-selection static --mask-channel-other 0 --mask-channel-length 64 --mask-channel-prob 0.5 --zero-infinity \
--feature-grad-mult 0.0 --freeze-finetune-updates 10000 --validate-after-updates 0 --optimizer adam \
--adam-betas '(0.9, 0.98)' --adam-eps 1e-08 --lr 5e-05 --lr-scheduler tri_stage --warmup-steps 10000 --hold-steps 90000 \
--decay-steps 100000 --final-lr-scale 0.05 --final-dropout 0.0 --dropout 0.0 --activation-dropout 0.1 --criterion bert_mlm \
--attention-dropout 0.0 --max-tokens 640000 --seed 2337 --log-format json --log-interval 500 --ddp-backend no_c10d \
--adapter-dimension 512 --decoder-bert-model-name $bert_model --tensorboard-logdir $output_path/events/$exp_namewav2bert_events/$exp_name \
--lexicon $bert_model/dict.multi.txt --only-bert --add-input
```
**rescore with finetune wav2vec2**
* w2l-decoder: BertRescore
* lm-batch-size: batch size when forwarding BERT
* beam: beam size for ctc beam search
* top-k-size: rescore for top-k size 
* score-lamda: weight for resocring the am and lm result 
* lm-model: BERT ckpt path
* path: wav2vec2 fine-tuned ckpt
```
python examples/speech_recognition/infer.py  manifest/202-swahili --task audio_pretraining \
--nbest 1 --path ckpts/swahili_base_1/checkpoint_best.pt --gen-subset eval --results-path ./result/wav2bert_results --w2l-decoder BertRescore \
--criterion ctc --labels ltr --max-tokens 640000 --lm-batch-size 10 --beam 10 \
--post-process letter  \
--top-k-size 10 --score-lamda 0.7 --decoder-bert-model-name ./bert_multi \
--lm-model ckpts/finetune_mbert_swahili_1/checkpoint_last.pt \
```

### plugin kenlm
**rescore with the kenlm language model**
* path: wav2vec2 ckpt
* w2l-decoder: kenlm
* lm-weight: kenlm score weight
* chinese-cer: calculate the cer instead of wer
* beam： beam search size
* lexicon: lexicon file spliting the word eg: <WORD   W O R D |>
```
python examples/speech_recognition/infer.py manifest/204-swahili --task audio_pretraining \
--nbest 1 --path ckpts/swahili_base_1/checkpoint_best.pt --gen-subset eval --results-path results/swahili_4gram_500beam --w2l-decoder kenlm \
--lm-model ./lm/swahili_lm/swahili-4-gram.arpa --lm-weight 0.87 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 640000 \
--post-process letter --lexicon ./lm/swahili_lm/swahili_lm_lexicon.txt --chinese-cer --beam 500 
```

# manifest 

differenet dataset has its own manifest meta file
eg：
```
--manifest/202-swahili
    --dict.ltr.txt          : dictionary vocab 
    --train.tsv             : tsv first line is audio dataset path，the format of the remaining line is: <audio_relative_path, audio_sample_point_num>
    --train.ltr             : text label in letter format，like <H E L L O | W O R L D>, using "|" to split words
    --train.wrd             : text label in word format，like <HELLO WORLD>
    --eval.tsv              : different split for test or validate
    --eval.ltr
    --eval.wrd
```

