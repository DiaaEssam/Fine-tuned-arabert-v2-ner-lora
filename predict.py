"""
Run Arabic NER inference on custom text.

Usage:
    python predict.py --model ./final_arabic_ner_lora --text "محمد يعمل في شركة جوجل في القاهرة"
    python predict.py --model ./final_arabic_ner_lora  # interactive mode
"""

import argparse
import json
import torch
from transformers import AutoTokenizer

from src.model import load_trained_model
from src.inference import predict_ner, extract_entities, format_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Arabic NER inference")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to saved model directory (merged or LoRA checkpoint)")
    parser.add_argument("--text", type=str, default=None,
                        help="Text to analyse (interactive mode if omitted)")
    parser.add_argument("--show-all", action="store_true",
                        help="Show all tokens, not just entities")
    return parser.parse_args()


def run(text: str, tokenizer, model, id_to_tag: dict, device, show_all: bool):
    print(f"\n📝 {text}")
    print("-" * 60)
    preds = predict_ner(text, tokenizer, model, id_to_tag, device)
    entities = extract_entities(preds)

    if entities:
        for etype, elist in entities.items():
            print(f"  [{etype}]  " + ",  ".join(elist))
    else:
        print("  No entities detected")

    if show_all or not entities:
        print(format_predictions(preds, show_all=True))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load label mappings
    with open(f"{args.model}/label_mappings.json", encoding="utf-8") as f:
        mappings = json.load(f)
    id_to_tag = {int(k): v for k, v in mappings["id2label"].items()}
    tag_to_id = mappings["label2id"]

    tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
    model = load_trained_model(args.model, id_to_tag, tag_to_id, device)

    if args.text:
        run(args.text, tokenizer, model, id_to_tag, device, args.show_all)
    else:
        print("Interactive mode — type Arabic text, or 'q' to quit.\n")
        while True:
            text = input(">>> ").strip()
            if text.lower() in ("q", "quit", "exit"):
                break
            if text:
                run(text, tokenizer, model, id_to_tag, device, args.show_all)


if __name__ == "__main__":
    main()
