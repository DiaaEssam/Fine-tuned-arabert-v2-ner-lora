"""
Inference utilities for Arabic NER — predict, format, and extract entities.
"""

import torch


def predict_ner(text: str, tokenizer, model, id_to_tag: dict, device) -> list:
    """
    Predict NER tags for a string of space-separated Arabic tokens.

    Args:
        text: Input text (space-separated Arabic tokens)
        tokenizer: HuggingFace tokenizer
        model: Trained model (merged or PEFT)
        id_to_tag: Mapping from label index to tag name
        device: torch.device

    Returns:
        List of (token, label, confidence) tuples
    """
    tokens = text.split()
    if not tokens:
        return []

    tokenized = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding=False,
    )
    word_ids = tokenized.word_ids(batch_index=0)
    inputs = {k: v.to(device) for k, v in tokenized.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0].cpu()
    probs = torch.softmax(logits, dim=-1)
    predictions = torch.argmax(logits, dim=-1).tolist()

    results = []
    previous_word_idx = None

    for idx, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx != previous_word_idx:
            if word_idx < len(tokens) and idx < len(predictions):
                pred_label = id_to_tag[predictions[idx]]
                confidence = probs[idx][predictions[idx]].item()
                results.append((tokens[word_idx], pred_label, confidence))
        previous_word_idx = word_idx

    return results


def extract_entities(predictions: list) -> dict:
    """
    Extract complete named entities from token-level predictions.

    Args:
        predictions: Output from predict_ner()

    Returns:
        Dictionary mapping entity type -> list of entity strings
    """
    entities: dict = {}
    current_entity: list = []
    current_type: str | None = None

    for token, label, _ in predictions:
        if label.startswith("B-"):
            if current_entity and current_type:
                entities.setdefault(current_type, []).append(" ".join(current_entity))
            current_type = label[2:]
            current_entity = [token]

        elif label.startswith("I-") and current_type == label[2:]:
            current_entity.append(token)

        else:
            if current_entity and current_type:
                entities.setdefault(current_type, []).append(" ".join(current_entity))
            current_entity = []
            current_type = None

    if current_entity and current_type:
        entities.setdefault(current_type, []).append(" ".join(current_entity))

    return entities


def format_predictions(predictions: list, show_all: bool = False) -> str:
    """
    Format token-level predictions as a human-readable string.

    Args:
        predictions: Output from predict_ner()
        show_all: If True, include non-entity tokens too

    Returns:
        Formatted string
    """
    if not predictions:
        return "No predictions"

    lines = [
        f"  • {token:<20} {label:<10} (conf: {confidence:.3f})"
        for token, label, confidence in predictions
        if show_all or label != "O"
    ]
    return "\n".join(lines) if lines else "  No entities detected"
