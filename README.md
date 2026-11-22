# HuBERT pretraining

## Speech-unit labels

Training does not fit a DBSCAN model or a local k-means model.
`train.py` uses `hubert_layer9_code500` from
[ASRP](https://pypi.org/project/asrp/) to generate discrete targets for both
Hugging Face audio datasets and custom audio files. This is the layer-9,
500-unit quantizer; the example below uses the corresponding
`--vocab_size 500`.

Install the project dependencies before running the training example:

```shell
pip install -r requirements.txt
```

## Training example

```shell
WANDB_MODE=disabled python train.py --tokenize_config facebook/hubert-large-ls960-ft --model_config \
ntu-spml/distilhubert --group_by_length --train_set hf-internal-testing/librispeech_asr_dummy --train_split validation \
--train_subset clean --test_split validation --test_subset clean --learning_rate 0.0003 --batch 30 --logging_steps 10 \
--eval_steps 60 --epoch 150 --use_auth_token True --output_dir ./model_test --overwrite_output_dir --batch 1 \
--vocab_size 500
```
