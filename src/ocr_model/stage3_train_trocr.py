"""
Stage 3: TrOCR Fine-tuning for Polish Handwriting

This script fine-tunes Microsoft's TrOCR model on the labeled handwriting dataset.
"""

import csv
import json
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)
from datasets import Dataset as HFDataset


@dataclass
class TrainingConfig:
    """Configuration for TrOCR fine-tuning."""
    model_name: str = "microsoft/trocr-small-handwritten"
    output_dir: str = "models/trocr-polish-handwriting"
    
    # Training hyperparameters
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # Data split
    train_split: float = 0.85
    
    # Saving
    save_steps: int = 200
    eval_steps: int = 200
    logging_steps: int = 50
    
    # Device
    fp16: bool = False  # Set True if GPU supports it
    
    # Seed for reproducibility
    seed: int = 42


def load_labeled_data(csv_path: Path, lines_dir: Path) -> List[Tuple[str, str]]:
    """Load labeled data from CSV, matching image paths."""
    data = []
    missing = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            img_name = row['image_name']
            transcription = row['transcription'].strip()
            
            # Parse image name: skan_026_page_001_preproc_line_001.png
            # -> data/lines/skan_026_page_001_preproc/line_001.png
            base_name = img_name.replace('.png', '')
            
            if '_line_' not in base_name:
                continue
            
            parts = base_name.rsplit('_line_', 1)
            folder = parts[0]
            line_num = parts[1]
            img_path = lines_dir / folder / f'line_{line_num}.png'
            
            if img_path.exists() and transcription:
                data.append((str(img_path), transcription))
            else:
                missing += 1
    
    print(f"Loaded {len(data)} labeled samples ({missing} missing/invalid)")
    return data


def create_dataset(
    data: List[Tuple[str, str]],
    processor: TrOCRProcessor,
    max_target_length: int = 128
) -> HFDataset:
    """Create HuggingFace dataset from image-text pairs."""
    
    def process_example(example):
        # Load and process image
        image = Image.open(example['image_path']).convert('RGB')
        pixel_values = processor(image, return_tensors='pt').pixel_values.squeeze()
        
        # Tokenize text
        labels = processor.tokenizer(
            example['text'],
            padding='max_length',
            max_length=max_target_length,
            truncation=True,
            return_tensors='pt'
        ).input_ids.squeeze()
        
        # Replace padding token id with -100 for loss calculation
        labels[labels == processor.tokenizer.pad_token_id] = -100
        
        return {
            'pixel_values': pixel_values,
            'labels': labels,
        }
    
    # Create dataset dict
    dataset_dict = {
        'image_path': [d[0] for d in data],
        'text': [d[1] for d in data],
    }
    
    dataset = HFDataset.from_dict(dataset_dict)
    dataset = dataset.map(process_example, remove_columns=['image_path', 'text'])
    
    return dataset


def compute_metrics(pred, processor):
    """Compute Character Error Rate (CER) metric."""
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    
    # pred_ids might be tuple from generate - take first element
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    
    # Filter out invalid token ids (negative values)
    import numpy as np
    pred_ids = np.where(pred_ids < 0, processor.tokenizer.pad_token_id, pred_ids)
    
    # Replace -100 with pad token in labels
    labels_ids = np.where(labels_ids == -100, processor.tokenizer.pad_token_id, labels_ids)
    
    # Decode predictions and labels  
    try:
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
    except Exception as e:
        # Fallback if decoding fails
        print(f"Warning: decode error: {e}")
        return {'cer': 1.0}
    
    # Calculate CER (simple character-level accuracy for now)
    total_chars = 0
    errors = 0
    for pred, label in zip(pred_str, label_str):
        total_chars += len(label)
        errors += sum(1 for p, l in zip(pred, label) if p != l)
        errors += abs(len(pred) - len(label))
    
    cer = errors / max(total_chars, 1)
    return {'cer': cer}


def main():
    config = TrainingConfig()
    
    # Set seed
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Paths
    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / 'data' / 'lines.csv'
    lines_dir = project_root / 'data' / 'lines'
    output_dir = project_root / config.output_dir
    
    print("=" * 60)
    print("TrOCR Fine-tuning for Polish Handwriting")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Loading labeled data...")
    all_data = load_labeled_data(csv_path, lines_dir)
    
    if len(all_data) < 10:
        print("ERROR: Not enough labeled data for training!")
        return
    
    # Shuffle and split
    random.shuffle(all_data)
    split_idx = int(len(all_data) * config.train_split)
    train_data = all_data[:split_idx]
    eval_data = all_data[split_idx:]
    
    print(f"Train samples: {len(train_data)}")
    print(f"Eval samples: {len(eval_data)}")
    
    # Load processor and model
    print(f"\n[2/5] Loading model: {config.model_name}")
    processor = TrOCRProcessor.from_pretrained(config.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(config.model_name)
    
    # Configure model for generation
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 128
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    
    # Create datasets
    print("\n[3/5] Preparing datasets...")
    train_dataset = create_dataset(train_data, processor)
    eval_dataset = create_dataset(eval_data, processor)
    
    # Training arguments
    print("\n[4/5] Setting up training...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        predict_with_generate=True,
        fp16=config.fp16,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        push_to_hub=False,
        report_to="none",  # Disable wandb etc
    )
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.tokenizer,
        data_collator=default_data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
    )
    
    # Train
    print("\n[5/5] Starting training...")
    print(f"Output directory: {output_dir}")
    trainer.train()
    
    # Save final model
    print("\n[DONE] Saving final model...")
    trainer.save_model(str(output_dir / 'final'))
    processor.save_pretrained(str(output_dir / 'final'))
    
    # Save config
    with open(output_dir / 'training_config.json', 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    print(f"\nModel saved to: {output_dir / 'final'}")
    print("Training complete!")


if __name__ == "__main__":
    main()
