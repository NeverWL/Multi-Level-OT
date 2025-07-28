#!/usr/bin/env python3
"""
Script to upload trained Multi-Level-OT models to HuggingFace Hub
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, login


def upload_model_to_hf(model_path, repo_name, token=None, private=False):
    """
    Upload a trained model to HuggingFace Hub
    
    Args:
        model_path: Path to the trained model directory
        repo_name: Name for the HuggingFace repository 
                  (e.g., "username/model-name")
        token: HuggingFace token (optional, will prompt if not provided)
        private: Whether to make the repository private
    """
    
    # Check if model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    # Login to HuggingFace
    if token:
        login(token=token)
    else:
        login()  # This will prompt for token if not already logged in
    
    api = HfApi()
    
    # Create repository
    try:
        api.create_repo(repo_name, private=private, exist_ok=True)
        print(f"✅ Repository {repo_name} created/verified")
    except Exception as e:
        print(f"⚠️  Repository creation warning: {e}")
    
    # Upload model files
    print(f"📤 Uploading model from {model_path} to {repo_name}...")
    
    # Upload all files in the model directory
    for file_path in Path(model_path).rglob("*"):
        if file_path.is_file():
            # Calculate relative path for upload
            relative_path = file_path.relative_to(model_path)
            remote_path = str(relative_path)
            
            print(f"  Uploading: {relative_path}")
            try:
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=remote_path,
                    repo_id=repo_name
                )
            except Exception as e:
                print(f"  ⚠️  Warning uploading {relative_path}: {e}")
    
    print(f"✅ Model uploaded successfully to: https://huggingface.co/{repo_name}")
    
    # Create model card
    create_model_card(repo_name, model_path)

def create_model_card(repo_name, model_path):
    """Create a basic model card for the uploaded model"""
    
    model_card_content = f"""---
language:
- en
license: mit
tags:
- knowledge-distillation
- multi-level-optimal-transport
- language-model
---

# {repo_name.split('/')[-1]}

This model was trained using Multi-Level Optimal Transport for knowledge distillation.

## Model Details

- **Base Model**: {Path(model_path).name}
- **Training Method**: Multi-Level Optimal Transport Distillation
- **Repository**: [Multi-Level-OT](https://github.com/your-repo/multi-level-ot)

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("{repo_name}")
model = AutoModelForCausalLM.from_pretrained("{repo_name}")

# Use the model for inference
inputs = tokenizer("Your input text here", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Details

This model was distilled using the Multi-Level Optimal Transport approach, which enables efficient knowledge transfer between models with different tokenizers.

## Citation

```bibtex
@article{{cui2024multi,
  title={{Multi-Level Optimal Transport for Universal Cross-Tokenizer Knowledge Distillation on Language Models}},
  author={{Cui, Xiao and Zhu, Mo and Qin, Yulei and Xie, Liang and Zhou, Wengang and Li, Houqiang}},
  journal={{arXiv preprint arXiv:2412.14528}},
  year={{2024}}
}}
"""
    
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=model_card_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_name
        )
        print("✅ Model card created")
    except Exception as e:
        print(f"⚠️  Warning creating model card: {e}")

def main():
    parser = argparse.ArgumentParser(description="Upload trained model to HuggingFace Hub")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to the trained model directory")
    parser.add_argument("--repo_name", type=str, required=True,
                       help="HuggingFace repository name (e.g., 'username/model-name')")
    parser.add_argument("--token", type=str, default=None,
                       help="HuggingFace token (optional)")
    parser.add_argument("--private", action="store_true",
                       help="Make the repository private")
    
    args = parser.parse_args()
    
    upload_model_to_hf(args.model_path, args.repo_name, args.token, args.private)

if __name__ == "__main__":
    main() 