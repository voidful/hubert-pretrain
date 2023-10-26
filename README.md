```shell
WANDB_MODE=disabled python train.py --tokenize_config facebook/hubert-large-ls960-ft --model_config \
ntu-spml/distilhubert --group_by_length --train_set hf-internal-testing/librispeech_asr_dummy --train_split validation \
--train_subset clean --test_split validation --test_subset clean --learning_rate 0.0003 --batch 30 --logging_steps 10 \
--eval_steps 60 --epoch 150 --use_auth_token True --output_dir ./model_test --overwrite_output_dir --batch 1 \
--vocab_size 500
```