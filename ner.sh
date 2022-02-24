#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

model_name=$1
dataset_name=$2
seed=$3

if [[ $dataset_name == 'pharmaconer' ]]; then
  train_file=$SCRIPT_DIR/datasets/pharmaconer/train.conll
  dev_file=$SCRIPT_DIR/datasets/pharmaconer/dev.conll
  test_file=$SCRIPT_DIR/datasets/pharmaconer/test.conll
  conll_loading_script=$SCRIPT_DIR/datasets/pharmaconer/pharmaconer.py
  
elif  [[ $dataset_name == 'cantemist' ]]; then
  train_file=$SCRIPT_DIR/datasets/cantemist/train.conll
  dev_file=$SCRIPT_DIR/datasets/cantemist/dev.conll
  test_file=$SCRIPT_DIR/datasets/cantemist/test.conll
  conll_loading_script=$SCRIPT_DIR/datasets/cantemist/cantemist.py
fi

python $SCRIPT_DIR/run_ner.py \
  --model_name_or_path $model_name \
  --train_file $train_file \
  --validation_file $dev_file \
  --test_file $test_file \
  --conll_loading_script $conll_loading_script \
  --do_train \
  --do_eval \
  --do_predict \
  --max_train_samples 10 \
  --max_eval_samples 10 \
  --max_predict_samples 10 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 10 \
  --load_best_model_at_end \
  --metric_for_best_model f1 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --overwrite_output_dir \
  --seed $seed \
  --logging_dir $SCRIPT_DIR/output/tb \
  --output_dir $SCRIPT_DIR/output 2>&1 | tee $SCRIPT_DIR/output/train.log
