#!/usr/bin/env python3
"""
amazon_polarity_loader.py — Text classification dataset loader with Amazon Polarity support.

Features:
- Loads HuggingFace 'amazon_polarity'
- Joins title + content -> text
- Tokenizes with a chosen tokenizer
- Builds PyTorch DataLoaders
- (Option B) Can SAVE the tokenized dataset to disk in several formats:
    --save-format hf|parquet|jsonl|pt
    --out-dir data/amazon_polarity_tok


  python dataset/amazon_polarity_loader.py \
    --model-name bert-base-uncased \
    --dataset amazon_polarity \
    --batch-size 32 \
    --max-length 256 \
    --num-workers 4 \
    --out-dir data/amazon_polarity_tok \
    --save-format hf
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

# -----------------------------
# Data config & helpers
# -----------------------------
@dataclass(frozen=True)
class TextKeys:
    text: str
    label: str


DATASET_SCHEMAS: Dict[str, TextKeys] = {
    "amazon_polarity": TextKeys(text="text", label="label"),
}

AMAZON_ID2LABEL = {0: "negative", 1: "positive"}
AMAZON_LABEL2ID = {v: k for k, v in AMAZON_ID2LABEL.items()}


def _load_raw_dataset(name: str, cache_dir: Optional[str] = None) -> DatasetDict:
    """
    Load a raw Hugging Face dataset as a DatasetDict with 'train' and 'validation'.
    For amazon_polarity: use the official 'train' and 'test' splits, mapping test->validation.
    """
    if name == "amazon_polarity":
        ds = load_dataset("amazon_polarity", cache_dir=cache_dir)
        return DatasetDict(train=ds["train"], validation=ds["test"])
    raise ValueError(f"Unsupported dataset '{name}'. Add it to _load_raw_dataset & DATASET_SCHEMAS.")


def _attach_text_field(ds: DatasetDict, dataset_name: str) -> DatasetDict:
    """
    Normalize to a single 'text' field that training loops expect.
    For amazon_polarity, concatenate title + content.
    """
    if dataset_name == "amazon_polarity":
        def _to_text(batch):
            title = batch.get("title", "") or ""
            content = batch.get("content", "") or ""
            text = (title.strip() + " " + content.strip()).strip()
            return {"text": text}

        return DatasetDict(
            train=ds["train"].map(_to_text, remove_columns=[]),
            validation=ds["validation"].map(_to_text, remove_columns=[]),
        )
    return ds


def build_tokenizer(model_name: str, use_fast: bool = True):
    return AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)


def _tokenize_dataset(
    ds: DatasetDict,
    tokenizer,
    text_key: str,
    max_length: int,
) -> DatasetDict:
    def tok_fn(batch):
        return tokenizer(batch[text_key], truncation=True, max_length=max_length)

    # compute remove_columns per split, keep only text + label for saving
    def keep_cols(split):
        cols = ds[split].column_names
        return [c for c in cols if c not in (text_key, "label")]

    tokenized = DatasetDict(
        train=ds["train"].map(tok_fn, batched=True, remove_columns=keep_cols("train")),
        validation=ds["validation"].map(tok_fn, batched=True, remove_columns=keep_cols("validation")),
    )
    return tokenized


def _maybe_limit(ds: DatasetDict, limit_train: Optional[int], limit_eval: Optional[int]) -> DatasetDict:
    def head(split, n):
        return split.select(range(min(n, len(split))))
    if limit_train:
        ds = DatasetDict(train=head(ds["train"], limit_train), validation=ds["validation"])
    if limit_eval:
        ds = DatasetDict(train=ds["train"], validation=head(ds["validation"], limit_eval))
    return ds


def prepare_dataloaders(
    model_name: str,
    dataset: str = "amazon_polarity",
    batch_size: int = 32,
    max_length: int = 256,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
    limit_train: Optional[int] = None,
    limit_eval: Optional[int] = None,
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,
    out_dir: Optional[str] = None,            # NEW: save path
    save_format: str = "hf",                  # NEW: save format
):
    """
    Returns:
      loaders: {'train': DataLoader, 'validation': DataLoader}
      tokenizer: HF tokenizer
      label_maps: {'id2label': {int:str}, 'label2id': {str:int}}
    """
    if dataset not in DATASET_SCHEMAS:
        raise ValueError(f"Dataset '{dataset}' is not registered. Available: {list(DATASET_SCHEMAS.keys())}")

    # 1) Load & normalize text
    raw = _load_raw_dataset(dataset, cache_dir=cache_dir)
    norm = _attach_text_field(raw, dataset)

    # 2) Tokenizer & tokenization
    tokenizer = build_tokenizer(model_name)
    schema = DATASET_SCHEMAS[dataset]
    tokenized = _tokenize_dataset(norm, tokenizer, text_key=schema.text, max_length=max_length)

    # 3) Optional subsetting
    tokenized = _maybe_limit(tokenized, limit_train=limit_train, limit_eval=limit_eval)

    # 4) Optional saving (BEFORE converting to torch format)
    if out_dir is not None:
        import os, json, torch as _torch
        os.makedirs(out_dir, exist_ok=True)
        if save_format == "hf":
            tokenized.save_to_disk(out_dir)
        elif save_format == "parquet":
            for split in ["train", "validation"]:
                tokenized[split].to_pandas().to_parquet(os.path.join(out_dir, f"{split}.parquet"), index=False)
        elif save_format == "jsonl":
            for split in ["train", "validation"]:
                with open(os.path.join(out_dir, f"{split}.jsonl"), "w", encoding="utf-8") as f:
                    for ex in tokenized[split]:
                        f.write(json.dumps({k: ex[k] for k in ex.keys()}) + "\n")
        elif save_format == "pt":
            cols = ["input_ids", "attention_mask", "label"]
            for split in ["train", "validation"]:
                data = {c: tokenized[split][c] for c in cols}
                _torch.save(data, os.path.join(out_dir, f"{split}.pt"))
        else:
            raise ValueError(f"Unknown save_format '{save_format}'")
        print(f"[INFO] Saved preprocessed dataset to: {out_dir} ({save_format})")

    # 5) Collator and PyTorch DataLoaders
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_torch = DatasetDict(
        train=tokenized["train"].with_format(type="torch", columns=["input_ids", "attention_mask", "label"]),
        validation=tokenized["validation"].with_format(type="torch", columns=["input_ids", "attention_mask", "label"]),
    )

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    train_loader = DataLoader(
        tokenized_torch["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        tokenized_torch["validation"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collator,
    )

    # 6) Label maps
    if dataset == "amazon_polarity":
        id2label = AMAZON_ID2LABEL
        label2id = AMAZON_LABEL2ID
    else:
        # generic fallback (unused here)
        unique = sorted(set(tokenized["train"]["label"].tolist() + tokenized["validation"]["label"].tolist()))
        id2label = {int(i): str(i) for i in unique}
        label2id = {v: k for k, v in id2label.items()}

    loaders = {"train": train_loader, "validation": val_loader}
    return loaders, tokenizer, {"id2label": id2label, "label2id": label2id}


# -----------------------------
# CLI for quick checks / export
# -----------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="Build DataLoaders with Amazon Polarity support.")
    p.add_argument("--model-name", type=str, default="bert-base-uncased")
    p.add_argument("--dataset", type=str, default="amazon_polarity", choices=list(DATASET_SCHEMAS.keys()))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-eval", type=int, default=None)
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None,
                   help="Optional: directory to write preprocessed dataset.")
    p.add_argument("--save-format", type=str, default="hf",
                   choices=["hf", "parquet", "jsonl", "pt"],
                   help="How to save: HuggingFace save_to_disk, Parquet, JSONL, or torch tensors.")
    return p.parse_args()


def main():
    args = _parse_args()
    loaders, tokenizer, label_maps = prepare_dataloaders(
        model_name=args.model_name,
        dataset=args.dataset,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        limit_train=args.limit_train,
        limit_eval=args.limit_eval,
        out_dir=args.out_dir,
        save_format=args.save_format,
    )
    # Simple smoke test: iterate one batch
    batch = next(iter(loaders["train"]))
    print("Tokenizer:", tokenizer.name_or_path)
    print("Label map:", label_maps["id2label"])
    print("Train batch tensors:", {k: v.shape for k, v in batch.items()})


if __name__ == "__main__":
    main()
