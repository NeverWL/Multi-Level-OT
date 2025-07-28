#export HOME = ""

export CUDA_VISIBLE_DEVICES=0 python finetuning.py \
--model_name $HOME/Multi-Level-OT/google/gemma-3-4b-pt \
--dataset.file $HOME/Multi-Level-OT/llm_distillation/datasets/loader/qed.py \
--lr 1e-6 \
--num_epochs 5 \
--batch_size_training 1 \
--val_batch_size 1 \
--output_dir $HOME/Multi-Level-OT/output2 \
--distillation_config_model_name $HOME/models/ModelSpace/GemmaX2-28-9B-v0.1 \
--distillation \
--distillation_config_enable_fsdp \
--distillation_config_pure_bf16 \
--distillation_config_distil_factor 1.5 \
--save_step 2000 \
--f 1
