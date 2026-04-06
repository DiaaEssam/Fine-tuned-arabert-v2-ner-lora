"""
Dataset loading and tokenization utilities for Arabic NER.
"""

from datasets import load_dataset, concatenate_datasets


DATASET_NAME = "iSemantics/conllpp-ner-ar"


def load_ner_dataset():
    """
    Load the Arabic NER dataset from HuggingFace Hub.

    Returns:
        dataset: DatasetDict with train/validation/test splits
        label_list: List of label names
        id_to_tag: Dict mapping int -> tag string
        tag_to_id: Dict mapping tag string -> int
    """
    dataset = load_dataset(DATASET_NAME)
    label_list = dataset["train"].features["ner_tags"].feature.names
    id_to_tag = {idx: label for idx, label in enumerate(label_list)}
    tag_to_id = {label: idx for idx, label in enumerate(label_list)}
    return dataset, label_list, id_to_tag, tag_to_id


def make_tokenize_fn(tokenizer, max_length: int = 128):
    """
    Returns a batched tokenization function that aligns NER labels with subword tokens.

    Subword tokens after the first and special tokens are assigned label -100
    so they are ignored during loss computation.

    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length

    Returns:
        tokenize_and_align_labels: Callable for use with Dataset.map()
    """

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding=False,
            max_length=max_length,
        )

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    return tokenize_and_align_labels


def tokenize_dataset(dataset, tokenizer, max_length: int = 128):
    """
    Tokenize all splits of the dataset.

    Args:
        dataset: DatasetDict from load_ner_dataset()
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length

    Returns:
        tokenized_datasets: DatasetDict with tokenized splits
    """
    tokenize_fn = make_tokenize_fn(tokenizer, max_length)
    return dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
        load_from_cache_file=False,
    )


def get_combined_train(tokenized_datasets):
    """Combine train and validation splits for final training."""
    return concatenate_datasets(
        [tokenized_datasets["train"], tokenized_datasets["validation"]]
    )
