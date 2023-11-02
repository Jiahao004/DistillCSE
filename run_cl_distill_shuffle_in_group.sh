#!/bin/bash


device=0
lr=3e-5
cl_temp=0.05
distill_weight=1
distill_temp1=0.025
distill_temp2=0.0125

# single teacher
distill_teacher="princeton-nlp/unsup-simcse-bert-base-uncased"

# multiple teacher components
#distill_teacher="princeton-nlp/unsup-simcse-bert-base-uncased ../SimCSE/result/unsup-simcse-seed5 ../SimCSE/result/unsup-simcse-seed42 ../SimCSE/result/unsup-simcse-seed47
#../SimCSE/result/unsup-simcse-seed2023 ../SimCSE/result/unsup-simcse-seed12 ../SimCSE/result/unsup-simcse-seed2 ../SimCSE/result/unsup-simcse-seed0
#../SimCSE/result/unsup-simcse-seed117 ../SimCSE/result/unsup-simcse-seed123 ../SimCSE/result/unsup-simcse-seed142 ../SimCSE/result/unsup-simcse-seed177
#../SimCSE/result/unsup-simcse-seed587 ../SimCSE/result/unsup-simcse-seed529 ../SimCSE/result/unsup-simcse-seed2345 ../SimCSE/result/unsup-simcse-seed139"




model=bert-base-uncased

echo teacher is $distill_teacher
group_size_by_prob=0.1

output_dir=result/$model-teacher-princeton-lr$lr-distill-$distill_weight-$distill_temp1-$distill_temp2-group_shuffling_by_prob$group_size_by_prob
echo output_dir is $output_dir
CUDA_VISIBLE_DEVICES=$device   \
python3 train_distill_calibrate.py --fp16 --max_steps 5000 --group_shuffling --group_size_by_prob $group_size_by_prob \
    --distill_teacher $distill_teacher --distill_weight $distill_weight --distill_temp1 $distill_temp1 --distill_temp2 $distill_temp2 \
    --model_name_or_path $model \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir $output_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate $lr \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp $cl_temp \
    --do_train
python3 evaluation.py --model_name_or_path $output_dir --pooler cls_before_pooler > $output_dir/eval.txt
cat $output_dir/eval.txt
