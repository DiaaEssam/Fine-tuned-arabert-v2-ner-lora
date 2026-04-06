# arabic-ner-arabertv2

Fine-tuning **AraBERTv2** for Arabic Named Entity Recognition (NER) using **LoRA** (Low-Rank Adaptation). Trained on the [CoNLL++ Arabic NER dataset](https://huggingface.co/datasets/iSemantics/conllpp-ner-ar).

🤗 **Model on Hugging Face:** [Diaa-Essam/arabert-v2-ner-lora](https://huggingface.co/Diaa-Essam/arabert-v2-ner-lora)

---

## Features

- **AraBERTv2** (`aubmindlab/bert-base-arabertv2`) as the backbone
- **LoRA** adapters (rank 32) — fine-tune only ~3% of parameters
- **Weighted cross-entropy loss** to handle class imbalance (MISC entities upweighted)
- **Label smoothing** + cosine LR scheduler
- Supports 4 entity types: `PER`, `ORG`, `LOC`, `MISC`

---

## Results

| Metric    | Score  |
|-----------|--------|
| Precision | 0.839822|
| Recall    | 0.861660      |
| F1        | 0.850601      |
| Accuracy  | 0.9460      |

---

## Project Structure

```
arabic-ner-arabertv2/
├── src/
│   ├── data.py          # Dataset loading & tokenization
│   ├── model.py         # Model, LoRA config, WeightedTrainer
│   ├── metrics.py       # seqeval metrics & reporting
│   └── inference.py     # predict_ner, extract_entities, format_predictions
├── notebooks/
│   └── arabic_ner_lora.ipynb   # Original Kaggle notebook
├── train.py             # End-to-end training script
├── evaluate.py          # Evaluate a saved checkpoint
├── predict.py           # Inference on custom Arabic text
├── requirements.txt
└── .gitignore
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train

```bash
python train.py
```

With custom hyperparameters:

```bash
python train.py \
  --epochs 70 \
  --lr 1e-4 \
  --batch-size 32 \
  --lora-r 32 \
  --lora-alpha 64 \
  --output-dir ./aner_lora_model \
  --final-dir ./final_arabic_ner_lora
```

### 3. Evaluate

```bash
python evaluate.py --checkpoint ./aner_lora_model/checkpoint-XXXXX
```

### 4. Predict

Single sentence:
```bash
python predict.py \
  --model ./final_arabic_ner_lora \
  --text "محمد يعمل في شركة جوجل في القاهرة"
```

Interactive mode:
```bash
python predict.py --model ./final_arabic_ner_lora
```

---

## Training Details

| Setting                | Value                      |
|------------------------|----------------------------|
| Base model             | `aubmindlab/bert-base-arabertv2` |
| Dataset                | `iSemantics/conllpp-ner-ar` |
| LoRA rank (r)          | 32                         |
| LoRA alpha             | 64                         |
| LoRA target modules    | query, key, value, dense   |
| Learning rate          | 1e-4                       |
| LR scheduler           | Cosine                     |
| Batch size             | 32                         |
| Epochs                 | 70                         |
| Label smoothing        | 0.1                        |
| Warmup ratio           | 0.15                       |
| Max sequence length    | 128                        |
| Mixed precision        | fp16 (if GPU available)    |

### Class Weights

The `O` tag is heavily majority; MISC entities are rare. The `WeightedTrainer` applies:

| Label  | Weight |
|--------|--------|
| B/I-MISC | 2.5  |
| O      | 0.5    |
| Others | 1.0    |

---

## Entity Types

| Tag  | Description         | Example (AR)         |
|------|---------------------|----------------------|
| PER  | Person names        | محمد، السيسي         |
| ORG  | Organizations       | جوجل، الأمم المتحدة |
| LOC  | Locations           | القاهرة، نيويورك    |
| MISC | Miscellaneous       | الأهلي، آيفون       |

---

## Dataset

[iSemantics/conllpp-ner-ar](https://huggingface.co/datasets/iSemantics/conllpp-ner-ar) — Arabic CoNLL++ NER dataset loaded automatically via the `datasets` library. No manual download required.

---

## License

MIT
