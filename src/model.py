"""
Model definition and LoRA configuration for Arabic NER.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForTokenClassification
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import Trainer


BASE_MODEL_NAME = "aubmindlab/bert-base-arabertv2"


def load_base_model(id_to_tag: dict, tag_to_id: dict):
    """Load the AraBERT base model for token classification."""
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=len(tag_to_id),
        id2label=id_to_tag,
        label2id=tag_to_id,
    )
    return model


def apply_lora(model, r: int = 32, lora_alpha: int = 64, lora_dropout: float = 0.05):
    """
    Apply LoRA adapters to the model.

    Args:
        model: Base HuggingFace model
        r: LoRA rank
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout probability on LoRA layers

    Returns:
        PEFT model with LoRA applied
    """
    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=["query", "value", "key", "dense"],
        inference_mode=False,
    )
    return get_peft_model(model, lora_config)


def load_trained_model(checkpoint_path: str, id_to_tag: dict, tag_to_id: dict, device):
    """
    Load a trained LoRA checkpoint, merge weights, and prepare for inference.

    Args:
        checkpoint_path: Path to the saved LoRA checkpoint directory
        id_to_tag: Mapping from label index to tag name
        tag_to_id: Mapping from tag name to label index
        device: torch device

    Returns:
        Merged model ready for inference
    """
    base_model = load_base_model(id_to_tag, tag_to_id)
    peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)
    merged_model = peft_model.merge_and_unload()
    merged_model.to(device)
    merged_model.eval()
    return merged_model


class WeightedTrainer(Trainer):
    """
    Custom Trainer with class weighting to handle label imbalance.
    Upweights MISC entities and downweights the 'O' tag.
    """

    # Weights correspond to: B-LOC, B-MISC, B-ORG, B-PER,
    #                         I-LOC, I-MISC, I-ORG, I-PER, O
    CLASS_WEIGHTS = [1.0, 2.5, 1.0, 1.0, 1.0, 2.5, 1.0, 1.0, 0.5]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        class_weights = torch.tensor(self.CLASS_WEIGHTS, device=logits.device)

        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
            weight=class_weights,
            ignore_index=-100,
        )

        return (loss, outputs) if return_outputs else loss
