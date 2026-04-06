"""
Evaluate a trained Arabic NER checkpoint on the test set.

Usage:
    python evaluate.py --checkpoint ./aner_lora_model/checkpoint-13430
"""

import argparse
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

from src.data import load_ner_dataset, tokenize_dataset
from src.model import load_trained_model
from src.metrics import (
    make_compute_metrics,
    get_predictions,
    print_classification_report,
    print_entity_stats,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Arabic NER checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to LoRA checkpoint directory")
    parser.add_argument("--max-length", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    print("\nLoading dataset...")
    dataset, _, id_to_tag, tag_to_id = load_ner_dataset()
    tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
    tokenized_datasets = tokenize_dataset(dataset, tokenizer, args.max_length)

    # Model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model = load_trained_model(args.checkpoint, id_to_tag, tag_to_id, device)

    # Trainer
    eval_args = TrainingArguments(
        output_dir="./eval_output",
        per_device_eval_batch_size=64,
        dataloader_num_workers=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=make_compute_metrics(id_to_tag),
    )

    # Evaluate
    print("\nEvaluating on test set...")
    results = trainer.evaluate(tokenized_datasets["test"])
    print(f"\n{'='*50}")
    print("TEST SET RESULTS")
    print(f"{'='*50}")
    print(f"Loss:      {results['eval_loss']:.4f}")
    print(f"Accuracy:  {results['eval_accuracy']:.4f}")
    print(f"Precision: {results['eval_precision']:.4f}")
    print(f"Recall:    {results['eval_recall']:.4f}")
    print(f"F1 Score:  {results['eval_f1']:.4f}")

    # Detailed report
    true_labels, true_preds = get_predictions(trainer, tokenized_datasets["test"], id_to_tag)
    print(f"\n{'='*50}")
    print("CLASSIFICATION REPORT")
    print(f"{'='*50}")
    print_classification_report(true_labels, true_preds)

    print(f"\n{'='*50}")
    print("ENTITY STATISTICS")
    print(f"{'='*50}")
    print_entity_stats(true_labels, true_preds)


if __name__ == "__main__":
    main()
