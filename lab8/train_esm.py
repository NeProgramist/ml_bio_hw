"""Train a small ESM-2 classifier for the Week 8 practical.

The task is multi-label protein subcellular-localization prediction.  The
script uses SwissProt partitions 0--3 for training, partition 4 for
validation, and the supplied HPA table as an external test set.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


TARGET_COLUMNS = [
    "Cytoplasm",
    "Nucleus",
    "Cell membrane",
    "Mitochondrion",
    "Endoplasmic reticulum",
    "Lysosome/Vacuole",
    "Golgi apparatus",
    "Peroxisome",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--hpa-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/esm35m_frozen"))
    parser.add_argument("--model-name", default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument(
        "--mode",
        choices=("frozen", "partial", "full"),
        default=None,
        help="Convenience alias; use --num-unfrozen-layers for explicit control.",
    )
    parser.add_argument(
        "--num-unfrozen-layers",
        type=int,
        default=None,
        help="0=frozen backbone, N=unfreeze the last N blocks, -1=full fine-tuning.",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="LoRA fine-tuning: freeze the backbone and train low-rank adapters + head.",
    )
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["query", "value"],
        help="ESM Linear layers to adapt: query key value attention.output.dense "
        "intermediate.dense output.dense",
    )
    parser.add_argument("--max-length", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Stop after this many epochs without val macro-F1 improvement; 0 disables.",
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Model state_dict to start from (e.g. best_model.pt of an interrupted run). "
        "The optimizer starts fresh.",
    )
    parser.add_argument(
        "--resume-history",
        type=Path,
        default=None,
        help="history.csv of the epochs already completed by the weights in --resume-from. "
        "Training continues at the next epoch, and early stopping counts from that history.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_num_unfrozen_layers(mode, num_unfrozen_layers):
    """Resolve the explicit layer count while preserving the mode shortcut."""
    mode_defaults = {"frozen": 0, "partial": 2, "full": -1}

    if num_unfrozen_layers is None:
        return mode_defaults[mode or "frozen"]

    if num_unfrozen_layers < -1:
        raise ValueError("--num-unfrozen-layers must be -1, 0, or a positive integer.")

    if mode is not None:
        expected = mode_defaults[mode]
        if mode == "partial" and num_unfrozen_layers > 0:
            return num_unfrozen_layers
        if num_unfrozen_layers != expected:
            raise ValueError(
                f"Conflicting options: --mode {mode!r} implies "
                f"{expected}, but --num-unfrozen-layers is {num_unfrozen_layers}."
            )

    return num_unfrozen_layers


def load_splits(data_path: Path, hpa_path: Path):
    swiss = pd.read_csv(data_path).rename(
        columns={"ACC": "protein_id", "Sequence": "sequence", "Partition": "partition"}
    )
    hpa = pd.read_csv(hpa_path).rename(columns={"sid": "protein_id", "fasta": "sequence"})

    train = swiss[swiss["partition"].isin([0, 1, 2, 3])].copy()
    val = swiss[swiss["partition"] == 4].copy()

    # Exclude rows whose only labels are categories absent from the HPA table.
    train = train[train[TARGET_COLUMNS].sum(axis=1) > 0].reset_index(drop=True)
    val = val[val[TARGET_COLUMNS].sum(axis=1) > 0].reset_index(drop=True)
    test = hpa.reset_index(drop=True)
    return train, val, test


class ProteinDataset(Dataset):
    def __init__(self, frame, tokenizer, label_columns, max_length):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label_columns = label_columns
        self.max_length = max_length

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        encoded = self.tokenizer(
            row["sequence"],
            truncation=True,
            max_length=self.max_length,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "label": torch.tensor(
                row[self.label_columns].to_numpy(dtype=np.float32), dtype=torch.float32
            ),
        }


def make_collate_fn(tokenizer):
    """Pad every batch only to the length of its longest protein."""

    def collate(batch):
        padded = tokenizer.pad(
            [
                {"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]}
                for item in batch
            ],
            padding="longest",
            return_tensors="pt",
        )
        return {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "label": torch.stack([item["label"] for item in batch]),
        }

    return collate


def masked_mean_pool(hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(hidden_state)
    return (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


class ProteinFoundationClassifier(nn.Module):
    def __init__(self, backbone, hidden_dim, n_labels):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(hidden_dim, n_labels)

    def forward(self, input_ids, attention_mask):
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = masked_mean_pool(output.last_hidden_state, attention_mask)
        return self.classifier(pooled)


def find_transformer_blocks(backbone):
    """Find the Transformer block list for common Hugging Face layouts."""
    candidates = [
        ("encoder.layer", getattr(getattr(backbone, "encoder", None), "layer", None)),
        (
            "esm.encoder.layer",
            getattr(getattr(getattr(backbone, "esm", None), "encoder", None), "layer", None),
        ),
        (
            "transformer.layer",
            getattr(getattr(backbone, "transformer", None), "layer", None),
        ),
    ]
    for path, blocks in candidates:
        if blocks is not None and len(blocks) > 0:
            return path, blocks
    raise RuntimeError(
        "Could not find Transformer blocks. Inspect the loaded backbone with print(backbone)."
    )


def configure_trainability(model, num_unfrozen_layers):
    """Configure 0=frozen, N=last N blocks, -1=full fine-tuning."""
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    if num_unfrozen_layers == -1:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = True
    elif num_unfrozen_layers > 0:
        _, layers = find_transformer_blocks(model.backbone)
        if num_unfrozen_layers > len(layers):
            raise ValueError(
                f"Requested {num_unfrozen_layers} blocks, but the model has {len(layers)}."
            )
        for layer in layers[-num_unfrozen_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )
    return model


def apply_lora(model, args):
    """Wrap the (already frozen) backbone with trainable LoRA adapters."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        task_type="FEATURE_EXTRACTION",
    )
    model.backbone = get_peft_model(model.backbone, config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"LoRA r={args.lora_r} alpha={args.lora_alpha} on {args.lora_target_modules}. "
        f"Trainable parameters (adapters + head): {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )
    return model


def collect_metrics(y_true, logits):
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "macro_f1": f1_score(y_true, predictions, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, predictions, average="micro", zero_division=0),
        "exact_match": accuracy_score(y_true, predictions),
    }


def run_epoch(model, loader, criterion, optimizer, device, training, return_outputs=False):
    model.train(training)
    if training and not any(p.requires_grad for p in model.backbone.parameters()):
        model.backbone.eval()

    total_loss = 0.0
    all_labels, all_logits = [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            if training:
                optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            all_labels.append(labels.detach().cpu().numpy())
            all_logits.append(logits.detach().cpu().numpy())

    all_labels, all_logits = np.vstack(all_labels), np.vstack(all_logits)
    metrics = collect_metrics(all_labels, all_logits)
    metrics["loss"] = total_loss / len(loader.dataset)
    if return_outputs:
        return metrics, all_labels, all_logits
    return metrics


def build_predictions_table(frame, y_true, logits, label_columns, threshold=0.5):
    """One row per protein: true labels, per-label probabilities and predictions."""
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    predictions = (probabilities >= threshold).astype(int)
    table = pd.DataFrame({"protein_id": frame["protein_id"].to_numpy()})
    for index, label in enumerate(label_columns):
        table[f"true_{label}"] = y_true[:, index].astype(int)
        table[f"prob_{label}"] = probabilities[:, index]
        table[f"pred_{label}"] = predictions[:, index]
    table["true_labels"] = [
        "; ".join(label for label, value in zip(label_columns, row) if value) for row in y_true
    ]
    table["predicted_labels"] = [
        "; ".join(label for label, value in zip(label_columns, row) if value)
        for row in predictions
    ]
    return table


def main():
    args = parse_args()
    args.num_unfrozen_layers = resolve_num_unfrozen_layers(
        args.mode, args.num_unfrozen_layers
    )
    if args.lora and args.num_unfrozen_layers != 0:
        raise ValueError("--lora trains adapters on a frozen backbone; drop --mode/--num-unfrozen-layers.")
    args.mode = (
        "lora"
        if args.lora
        else "frozen"
        if args.num_unfrozen_layers == 0
        else "full"
        if args.num_unfrozen_layers == -1
        else "partial"
    )
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, val, test = load_splits(args.data_path, args.hpa_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    backbone = AutoModel.from_pretrained(args.model_name)
    model = ProteinFoundationClassifier(
        backbone, backbone.config.hidden_size, len(TARGET_COLUMNS)
    ).to(device)
    configure_trainability(model, args.num_unfrozen_layers)
    if args.lora:
        model = apply_lora(model, args).to(device)

    collate_fn = make_collate_fn(tokenizer)
    train_loader = DataLoader(
        ProteinDataset(train, tokenizer, TARGET_COLUMNS, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        ProteinDataset(val, tokenizer, TARGET_COLUMNS, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        ProteinDataset(test, tokenizer, TARGET_COLUMNS, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    positive = train[TARGET_COLUMNS].sum().to_numpy()
    pos_weight = (len(train) - positive) / (positive + 1e-6)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
    )
    default_learning_rates = {"frozen": 1e-3, "lora": 3e-4}
    learning_rate = args.learning_rate or default_learning_rates.get(args.mode, 2e-5)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )

    best_val_f1 = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    start_epoch = 1
    if args.resume_from is not None:
        # Load before the loop: the file may be the same best_model.pt this run overwrites.
        model.load_state_dict(torch.load(args.resume_from, map_location=device))
        print(f"Resumed model weights from {args.resume_from} (fresh optimizer state).")
    if args.resume_history is not None:
        history = pd.read_csv(args.resume_history).to_dict("records")
        best_row = max(history, key=lambda row: row["val_macro_f1"])
        best_val_f1, best_epoch = best_row["val_macro_f1"], int(best_row["epoch"])
        start_epoch = int(history[-1]["epoch"]) + 1
        epochs_without_improvement = start_epoch - 1 - best_epoch
        print(
            f"Continuing at epoch {start_epoch}: best val macro-F1 {best_val_f1:.4f} "
            f"at epoch {best_epoch}."
        )

    for epoch in range(start_epoch, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, False)
        row = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            "peak_gpu_memory_gb": (
                torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda"
                else None
            ),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(row)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.output_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping at epoch {epoch}: no val macro-F1 improvement "
                    f"for {args.patience} epochs (best epoch {best_epoch})."
                )
                break

    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    (args.output_dir / "labels.json").write_text(json.dumps(TARGET_COLUMNS, indent=2))
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "device": str(device),
                "train_size": len(train),
                "val_size": len(val),
                "test_size": len(test),
                "best_val_epoch": best_epoch,
                "epochs_run": len(history),
                "peak_gpu_memory_gb": (
                    max(row["peak_gpu_memory_gb"] for row in history)
                    if device.type == "cuda"
                    else None
                ),
            },
            default=str,
            indent=2,
        )
    )

    model.load_state_dict(torch.load(args.output_dir / "best_model.pt", map_location=device))
    test_metrics, test_labels, test_logits = run_epoch(
        model, test_loader, criterion, optimizer, device, training=False, return_outputs=True
    )
    (args.output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    build_predictions_table(test, test_labels, test_logits, TARGET_COLUMNS).to_csv(
        args.output_dir / "test_predictions.csv", index=False
    )
    print({"external_hpa_test": test_metrics})


if __name__ == "__main__":
    main()
