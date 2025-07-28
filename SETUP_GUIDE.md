# Multi-Level-OT Test Run Setup Guide

This guide will help you set up a test run using **ModelSpace/GemmaX2-28-9B-v0.1** as teacher and **google/gemma-3-4b-pt** as student on the **QED dataset**.

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended: A100 or similar)
- At least 16GB GPU memory for teacher model
- HuggingFace account with access to Llama-2 models

## Step 1: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Or install individually if you prefer:
pip install torch transformers datasets accelerate peft bitsandbytes
pip install sentencepiece protobuf numpy scipy scikit-learn
pip install rouge-score bert-score tqdm wandb tensorboard
pip install huggingface-hub safetensors tokenizers
```

## Step 2: Download Teacher Model (GemmaX2-28-9B-v0.1)

```bash
# Create models directory
mkdir -p $HOME/models

# Download GemmaX2-28-9B-v0.1
git clone https://huggingface.co/ModelSpace/GemmaX2-28-9B-v0.1 $HOME/models/ModelSpace/GemmaX2-28-9B-v0.1
```

**Note**: You may need to request access to this model first. Visit the HuggingFace page and click "Request access" if needed.

## Step 3: Download Student Model (Gemma-3-4B-PT)

```bash
# Create google directory
mkdir -p $HOME/Multi-Level-OT/google

# Download Gemma-3-4B-PT
git clone https://huggingface.co/google/gemma-3-4b-pt $HOME/Multi-Level-OT/google/gemma-3-4b-pt
```

## Step 4: Download Dataset

The QED dataset is available on Google Drive. Download it from:
https://drive.google.com/drive/folders/1ZE_wu0Ey2KpKrjq3NA0VgAvyhynOR6a4?usp=sharing

Extract and place it in:
```
$HOME/Multi-Level-OT/llm_distillation/datasets/processed/qed/
```

## Step 5: Verify Setup

Run the setup verification script:

```bash
chmod +x test_run_setup.sh
./test_run_setup.sh
```

This script will:
- Check if all required models and datasets are present
- Create output directory
- Run a test distillation

## Step 6: Manual Test Run (Alternative)

If you prefer to run manually:

```bash
export CUDA_VISIBLE_DEVICES=0
export HOME="/Users/jaredlim"  # Adjust to your home directory

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
```

## Step 7: Monitor Training

The training will:
- Save checkpoints every 100 steps
- Use FSDP for efficient distributed training
- Use BF16 precision for memory efficiency
- Apply Multi-Level Optimal Transport distillation

## Step 8: Evaluate Results

After training, evaluate the distilled model:

```bash
# Use the results.sh script to evaluate
export CUDA_VISIBLE_DEVICES=0 python $HOME/Multi-Level-OT/llm_distillation/benchmark/benchmark.py \
    --model_id "$HOME/Multi-Level-OT/test_output" \
    --model_tokenizer "$HOME/Multi-Level-OT/google/gemma-3-4b-pt" \
    --dataset_id "$HOME/Multi-Level-OT/llm_distillation/datasets/processed/qed" \
    --split_name "validation" \
    --context \
    --title \
    --batch_size 1 \
    --num_workers 1 \
    --output_path "$HOME/Multi-Level-OT/eval_results/" \
    --number_few_shot 0 \
    --context_length 1024 \
    --from_disk \
    --task "qa" \
    --save_predictions
```

## Step 9: Upload Model to HuggingFace Hub

**Yes, it's very easy to save your trained model to HuggingFace!** Here are several ways:

### Option 1: Using the provided upload script

```bash
# First, get your HuggingFace token from: https://huggingface.co/settings/tokens
python upload_to_hf.py \
    --model_path "$HOME/Multi-Level-OT/test_output" \
    --repo_name "your-username/gemma-3-4b-qed-distilled" \
    --token "your_hf_token_here"
```

### Option 2: Manual upload using transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# Login to HuggingFace
login(token="your_hf_token_here")

# Load your trained model
model = AutoModelForCausalLM.from_pretrained("$HOME/Multi-Level-OT/test_output")
tokenizer = AutoTokenizer.from_pretrained("$HOME/Multi-Level-OT/google/gemma-3-4b-pt")

# Upload to HuggingFace
model.push_to_hub("your-username/gemma-3-4b-qed-distilled")
tokenizer.push_to_hub("your-username/gemma-3-4b-qed-distilled")
```

### Option 3: Using the HuggingFace CLI

```bash
# Install huggingface_hub CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Upload the model directory
huggingface-cli upload your-username/gemma-3-4b-qed-distilled $HOME/Multi-Level-OT/test_output
```

### Option 4: Direct from training script

You can also modify the training script to automatically upload after training by adding this to the end of `train_utils.py`:

```python
# After training completes, upload to HuggingFace
if rank == 0 and train_config.upload_to_hf:
    model.student.push_to_hub("your-username/model-name")
```

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **Model Access Denied**: Request access to Llama-2 models from Meta
3. **Dataset Not Found**: Download from the provided Google Drive link
4. **FSDP Issues**: Try running without `--distillation_config_enable_fsdp` for single GPU
5. **HuggingFace Upload Issues**: Make sure you're logged in and have write permissions

### Memory Requirements:

- **Teacher Model**: ~28GB GPU memory (GemmaX2-28-9B)
- **Student Model**: ~8GB GPU memory (Gemma-3-4B)
- **Total**: ~36GB+ recommended

### Performance Tips:

- Use smaller batch sizes for limited GPU memory
- Enable gradient checkpointing for memory efficiency
- Use mixed precision training (already enabled with BF16)

## Expected Output

The training will create:
- Model checkpoints in `$HOME/Multi-Level-OT/test_output/`
- Training logs and metrics
- Final distilled model ready for evaluation
- Option to upload to HuggingFace Hub for easy sharing and deployment

## Next Steps

After successful test run:
1. Try different datasets (DialogSum, FairytaleQA)
2. Experiment with different teacher/student combinations
3. Adjust hyperparameters for better performance
4. Scale up to larger models or longer training
5. Share your models on HuggingFace Hub for the community! 