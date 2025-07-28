#!/bin/bash

# Test Run Setup for Multi-Level-OT
# Teacher: ModelSpace/GemmaX2-28-9B-v0.1
# Student: google/gemma-3-4b-pt
# Dataset: QED (Question Answering)

echo "Setting up test run for Multi-Level-OT..."

# Set environment variables
export HOME="/Users/jaredlim"
export CUDA_VISIBLE_DEVICES=0

# Check if required directories exist
echo "Checking required directories..."

# Check for teacher model
if [ ! -d "$HOME/models/ModelSpace/GemmaX2-28-9B-v0.1" ]; then
    echo "❌ Teacher model not found at $HOME/models/ModelSpace/GemmaX2-28-9B-v0.1"
    echo "Please download from: https://huggingface.co/ModelSpace/GemmaX2-28-9B-v0.1"
    echo "You may need to request access first."
    exit 1
else
    echo "✅ Teacher model found: GemmaX2-28-9B-v0.1"
fi

# Check for student model
if [ ! -d "$HOME/Multi-Level-OT/google/gemma-3-4b-pt" ]; then
    echo "❌ Student model not found at $HOME/Multi-Level-OT/google/gemma-3-4b-pt"
    echo "Please download from: https://huggingface.co/google/gemma-3-4b-pt"
    echo "Run: git clone https://huggingface.co/google/gemma-3-4b-pt $HOME/Multi-Level-OT/google/gemma-3-4b-pt"
    exit 1
else
    echo "✅ Student model found: Gemma-3-4B-PT"
fi

# Check for dataset
if [ ! -d "$HOME/Multi-Level-OT/llm_distillation/datasets/processed/qed" ]; then
    echo "❌ QED dataset not found"
    echo "Please download from: https://drive.google.com/drive/folders/1ZE_wu0Ey2KpKrjq3NA0VgAvyhynOR6a4?usp=sharing"
    echo "And place in: $HOME/Multi-Level-OT/llm_distillation/datasets/processed/qed"
    exit 1
else
    echo "✅ QED dataset found"
fi

# Create output directory
mkdir -p "$HOME/Multi-Level-OT/test_output"

echo ""
echo "🚀 Starting test run..."
echo "Teacher: GemmaX2-28-9B-v0.1"
echo "Student: Gemma-3-4B-PT"
echo "Dataset: QED"
echo "Output: $HOME/Multi-Level-OT/test_output"
echo ""

# Run the distillation
python finetuning.py \
    --model_name "$HOME/Multi-Level-OT/google/gemma-3-4b-pt" \
    --dataset.file "$HOME/Multi-Level-OT/llm_distillation/datasets/loader/qed.py" \
    --lr 1e-6 \
    --num_epochs 2 \
    --batch_size_training 1 \
    --val_batch_size 1 \
    --output_dir "$HOME/Multi-Level-OT/test_output" \
    --distillation_config_model_name "$HOME/models/ModelSpace/GemmaX2-28-9B-v0.1" \
    --distillation \
    --distillation_config_enable_fsdp \
    --distillation_config_pure_bf16 \
    --distillation_config_distil_factor 1.5 \
    --save_step 100 \
    --f 1

echo ""
echo "✅ Test run completed!"
echo "Check results in: $HOME/Multi-Level-OT/test_output" 