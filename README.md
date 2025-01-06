# Uncertainty-Aware-Contrastive-Learning-With-Hard-Negative-Sampling-for-Code-Search-Tasks


## Code Search Dataset
We conducted experiments on the CodeSearchNet code corpus.
Data statistic about the cleaned dataset for code document generation is shown in this Table.

| PL         | Training |  Dev   |  Test  | Candidates code |
| :--------- | :------: | :----: | :----: | :-------------: |
| Python     | 251,820  | 13,914 | 14,918 |     43,827      |
| PHP        | 241,241  | 12,982 | 14,014 |     52,660      |
| Go         | 167,288  | 7,325  | 8,122  |     28,120      |
| Java       | 164,923  | 5,183  | 10,955 |     40,347      |
| JavaScript |  58,025  | 3,885  | 3,291  |     13,981      |
| Ruby       |  24,927  | 1,400  | 1,261  |      4,360      |

### You can download and preprocess data from the publicly available datasets published by former study. [here](https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT/codesearch)


## run_cocosoda
```shell
lang=go
current_time=$(date "+%Y%m%d%H%M%S")

code_length=256
nl_length=128

model_type=base
lr=8e-6

batch_size=128
max_steps=1000
save_steps=100

base_model=
epoch=10

function fine-tune () {
output_dir=./saved_model/fine_tune/${lang}
mkdir -p $output_dir
echo ${output_dir}
 python run_cocosoda.py   --eval_frequency  100 \
    --output_dir ${output_dir}  \
    --config_name=${base_model}  \
    --model_name_or_path=${base_model} \
    --tokenizer_name=${base_model} \
    --lang=$lang \
    --do_train \
    --do_test \
    --train_data_file=./dataset/${lang}/train.jsonl \
    --eval_data_file=./dataset/${lang}/valid.jsonl \
    --test_data_file=./dataset/${lang}/test.jsonl \
    --codebase_file=./dataset/${lang}/codebase.jsonl \
    --neg_data_file=./dataset/${lang}/neg.jsonl \
    --num_train_epochs ${epoch} \
    --code_length ${code_length} \
    --nl_length ${nl_length} \
    --train_batch_size ${batch_size} \
    --eval_batch_size 64 \
    --learning_rate ${lr} \
    --seed x 2>&1| tee ${output_dir}/running.log
}
fine-tune

```
