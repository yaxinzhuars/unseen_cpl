#!/bin/sh
export GLUE_DIR='../datasets/NPTEL MOOC Dataset/train_test_5fold/'
export WIKI='../datasets/wiki_dpr/'
export MOOC='../mooc17/ML/'

# PYTHONPATH=../ CUDA_VISIBLE_DEVICES=0 python bert_pointwise.py \
# --task_name mooc \
# --do_train \
# --do_lower_case \
# --data_dir ../datasets/wiki_dpr/ \
# --bert_model bert-base-cased \
# --max_seq_length 48 \
# --train_batch_size 8 \
# --learning_rate 5e-5 \
# --num_train_epochs 10.0 \
# --output_dir /gypsum/scratch1/yaxinzhu/bert_output/uc_unseen/ \
# --old_output_dir /gypsum/scratch1/yaxinzhu/bert_output/lb_unseen_256/ \
# --data_prefix 'uc' \
# --data_postfix 'unseen' \
# --retrieval_augmentation  \
# --training_data output_uc_0.txt

PYTHONPATH=../ CUDA_VISIBLE_DEVICES=0 python bert_pointwise.py \
--task_name mooc \
--do_eval \
--do_lower_case \
--data_dir ../datasets/wiki_dpr/ \
--bert_model bert-base-cased \
--max_seq_length 48 \
--train_batch_size 8 \
--learning_rate 5e-5 \
--num_train_epochs 10.0 \
--output_dir /gypsum/scratch1/yaxinzhu/bert_output/uc_unseen/ \
--old_output_dir /gypsum/scratch1/yaxinzhu/bert_output/lb_unseen_256/ \
--data_prefix 'uc' \
--data_postfix 'unseen' \
--retrieval_augmentation  \
--training_data output_uc_0.txt

# PYTHONPATH=../ CUDA_VISIBLE_DEVICES=0 python bert_t5.py \
# --task_name mooc \
# --do_train \
# --do_lower_case \
# --data_dir ../datasets/wiki_dpr/ \
# --bert_model bert-base-cased \
# --max_seq_length 256 \
# --train_batch_size 8 \
# --learning_rate 5e-5 \
# --num_k 40 \
# --num_self_train 1 \
# --num_train_epochs 20.0 \
# --output_dir /gypsum/scratch1/yaxinzhu/bert_output/uc_t5/ \
# --old_output_dir /gypsum/scratch1/yaxinzhu/bert_output/uc_t5/ \
# --training_data output_uc_ttest1.txt \
# --seed 28

# python read_output.py \
# --c2w ../bert/uc_c2w.txt \
# --logits_file /gypsum/home/yaxinzhu/concept_g/bert/output_uc_unseen_0.txt \
# --input_train_file ../datasets/wiki_dpr/uc_train_1.txt \
# --input_test_file ../datasets/wiki_dpr/uc_test_1.txt \
# --output_file uc_train_mix_boost.index \
# --mode i

# python merge_data.py \
# --file1 ../datasets/name/uc_train_1.index \
# --file2 uc_train_mix_boost.index \
# --outfile uc_train_unseen_1.index