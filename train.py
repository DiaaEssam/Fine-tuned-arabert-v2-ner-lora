"""
Train Arabic NER model using AraBERTv2 + LoRA.

Usage:
    python train.py
    python train.py --epochs 50 --lr 2e-4 --batch-size 16 --output-dir ./my_model
"""

import argparse
import json
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)

from src.data import load_ner_dataset, tokenize_dataset, get_combined_train
from src.model import apply_lora, load_base_model, WeightedTrainer
from src.metrics import make_compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train Arabic NER with LoRA")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default="./aner_lora_model")
    parser.add_argument("--final-dir", type=str, default="./final_arabic_ner_lora")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    print("\nLoading dataset...")
    dataset, label_list, id_to_tag, tag_to_id = load_ner_dataset()
    print(f"Labels: {label_list}")

    tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
    tokenized_datasets = tokenize_dataset(dataset, tokenizer, args.max_length)
    combined_train = get_combined_train(tokenized_datasets)

    print(f"Training samples: {len(combined_train)}")
    print(f"Test samples:     {len(tokenized_datasets['test'])}")

    # Model
    print("\nLoading model and applying LoRA...")
    model = load_base_model(id_to_tag, tag_to_id)
    model = apply_lora(model, r=args.lora_r, lora_alpha=args.lora_alpha)
    model.to(device)
    model.print_trainable_parameters()

    # Training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.15,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=10,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        label_names=["labels"],
        label_smoothing_factor=0.1,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=combined_train,
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=make_compute_metrics(id_to_tag),
    )

    print("\n🚀 Starting training...\n")
    result = trainer.train()
    print(f"\nTraining loss:   {result.training_loss:.4f}")
    print(f"Training time:   {result.metrics['train_runtime'] / 60:.1f} min")

    # Save
    trainer.save_model(args.final_dir)
    tokenizer.save_pretrained(args.final_dir)
    with open(f"{args.final_dir}/label_mappings.json", "w", encoding="utf-8") as f:
        json.dump({"id2label": id_to_tag, "label2id": tag_to_id}, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Model saved to: {args.final_dir}")


if __name__ == "__main__":
    main()
