

CUDA_VISIBLE_DEVICES=3 python3 baseline.py \
	--model_type electra \
	--model_name_or_path /SISDC_GPFS/Home_SE/hy-suda/pre-train_model/electra-large/ \
	--do_train \
	--do_eval \
    --data_dir /SISDC_GPFS/Home_SE/hy-suda/lyl/MPDM/molweni/data \
	--train_file train.json \
	--dev_file dev.json \
    --test_file test.json \
	--output_dir ./output/baseline-electra2/ \
	--overwrite_output_dir \
	--per_gpu_train_batch_size 12 \
	--gradient_accumulation_steps 1 \
	--per_gpu_eval_batch_size 8 \
	--num_train_epochs 3 \
	--learning_rate 1e-5 \
    --threads 20 \
	--seed 42 \
	--do_lower_case \
    --evaluate_during_training \
	--max_answer_length 30 \
	--weight_decay 0.01 \
    --max_seq_length 384 \
	--fp16 --fp16_opt_level "O2" 
