"""
Evaluation metrics for Arabic NER using seqeval.
"""

import numpy as np
from collections import Counter
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


def make_compute_metrics(id_to_tag: dict):
    """
    Returns a compute_metrics function compatible with HuggingFace Trainer.

    Args:
        id_to_tag: Mapping from label index to tag name

    Returns:
        compute_metrics: Callable that takes (predictions, labels) and returns a dict
    """

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_labels = [
            [id_to_tag[l] for l in label if l != -100] for label in labels
        ]
        true_predictions = [
            [id_to_tag[pred] for (pred, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }

    return compute_metrics


def get_predictions(trainer, dataset, id_to_tag: dict):
    """
    Run inference and return aligned true labels and predictions.

    Args:
        trainer: HuggingFace Trainer instance
        dataset: Tokenized dataset split to evaluate
        id_to_tag: Mapping from label index to tag name

    Returns:
        true_labels: List of lists of true tag strings
        true_predictions: List of lists of predicted tag strings
    """
    predictions, labels, _ = trainer.predict(dataset)
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [id_to_tag[l] for l in label if l != -100] for label in labels
    ]
    true_predictions = [
        [id_to_tag[pred] for (pred, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions


def print_classification_report(true_labels, true_predictions):
    """Print a detailed per-entity classification report."""
    print(classification_report(true_labels, true_predictions, digits=4))


def count_entities(label_sequences):
    """Count entity mentions by type (using B- tags only)."""
    entity_counts = Counter()
    for sequence in label_sequences:
        for label in sequence:
            if label.startswith("B-"):
                entity_counts[label[2:]] += 1
    return entity_counts


def print_entity_stats(true_labels, true_predictions):
    """Print a comparison table of true vs predicted entity counts."""
    true_counts = count_entities(true_labels)
    pred_counts = count_entities(true_predictions)

    print(f"\n{'Entity':<15} {'True':>10} {'Pred':>10} {'Diff':>10}")
    print("-" * 50)

    for entity in sorted(set(list(true_counts) + list(pred_counts))):
        tc = true_counts.get(entity, 0)
        pc = pred_counts.get(entity, 0)
        print(f"{entity:<15} {tc:>10} {pc:>10} {pc - tc:>+10}")

    print("-" * 50)
    print(
        f"{'TOTAL':<15} {sum(true_counts.values()):>10} {sum(pred_counts.values()):>10}"
    )
