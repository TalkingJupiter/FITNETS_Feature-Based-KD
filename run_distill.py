#!/usr/bin/env python3
"""
Feature-based KD (FitNets-style) + response KD on Amazon Polarity (Option B: load dataset from disk).
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

# --- Telemetry (optional internal monitor) ---
class _NoOpMon:
    def __init__(self, *args, **kwargs): 
        pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def log_once(self): pass

try:
    # internal CSV monitor (context manager with .log_once())
    from monitor.monitor import PowerMonitor as _InternalMon
except Exception:
    _InternalMon = None


# -----------------------------
# Losses
# -----------------------------
def kd_loss_logits(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    ce = F.cross_entropy(student_logits, labels)
    log_p_s = F.log_softmax(student_logits / T, dim=-1)
    p_t = F.softmax(teacher_logits / T, dim=-1)
    kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
    total = alpha * ce + (1.0 - alpha) * kl
    return total, ce, kl


class Connector(nn.Module):
    """Map student feature dim -> teacher feature dim; identity if same."""
    def __init__(self, in_dim, out_dim, use_ln: bool = False):
        super().__init__()
        self.proj = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim) if use_ln else nn.Identity()
    def forward(self, x): return self.ln(self.proj(x))


def feature_loss(student_feat, teacher_feat, normalize=True):
    if normalize:
        student_feat = F.normalize(student_feat, dim=-1)
        teacher_feat = F.normalize(teacher_feat, dim=-1)
    return F.mse_loss(student_feat, teacher_feat)


# -----------------------------
# Data (Option B: from disk)
# -----------------------------
def build_loaders_from_disk(data_dir, tokenizer_name, batch_size, num_workers):
    """
    Loads a tokenized HF dataset from disk and ensures a 'labels' column exists.
    Accepts datasets that saved 'label' or 'labels' and normalizes to 'labels'.
    Also tolerates 'test' instead of 'validation'.
    """
    ds = load_from_disk(data_dir)  # expects tokenized columns: input_ids, attention_mask, label|labels

    # normalize label column name to 'labels'
    for split in ds.keys():
        cols = ds[split].column_names
        if "labels" not in cols and "label" in cols:
            ds = ds.rename_column("label", "labels")
            break

    val_split = "validation" if "validation" in ds else ("test" if "test" in ds else None)
    if val_split is None:
        raise ValueError("No 'validation' or 'test' split found in the dataset on disk.")

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    collator = DataCollatorWithPadding(tok)

    # ensure PyTorch tensors for keys we use
    ds = ds.with_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_loader = DataLoader(
        ds["train"],
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0),
        collate_fn=collator,
    )
    val_loader = DataLoader(
        ds[val_split],
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0),
        collate_fn=collator,
    )
    id2label = {0: "negative", 1: "positive"}
    label2id = {"negative": 0, "positive": 1}
    return {"train": train_loader, "validation": val_loader}, tok, {"id2label": id2label, "label2id": label2id}


# -----------------------------
# Training
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, required=True, help="Path to HF dataset (save_to_disk).")
    ap.add_argument("--output-dir", type=str, default="logs_fkd")
    ap.add_argument("--teacher-name", type=str, default="bert-base-uncased")
    ap.add_argument("--student-name", type=str, default="distilbert-base-uncased")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    # KD (logits)
    ap.add_argument("--kd-T", type=float, default=2.0)
    ap.add_argument("--kd-alpha", type=float, default=0.5, help="Weight on CE; 1-alpha on KL.")
    # Feature KD (FitNets)
    ap.add_argument("--feat-beta", type=float, default=1.0, help="Weight for feature loss.")
    ap.add_argument("--t-layer", type=int, default=-2, help="Teacher hidden_states index (e.g., -2 penultimate).")
    ap.add_argument("--s-layer", type=int, default=-2, help="Student hidden_states index.")
    ap.add_argument("--feat-pool", type=str, default="cls", choices=["cls", "mean"],
                    help="Pool token features into a vector.")
    ap.add_argument("--feat-ln", action="store_true", help="Use LayerNorm in connector.")
    ap.add_argument("--feat-normalize", action="store_true",
                    help="L2-normalize features before MSE.")
    ap.add_argument("--telemetry-every", type=int, default=1, help="Call monitor.log_once() every N train steps.")
    return ap.parse_args()


def pool_tokens(hidden, attention_mask, how="cls"):
    # hidden: [B, L, D]
    if how == "cls":
        return hidden[:, 0, :]
    # mean over valid tokens
    mask = attention_mask.unsqueeze(-1)           # [B, L, 1]
    summed = (hidden * mask).sum(dim=1)           # [B, D]
    denom = mask.sum(dim=1).clamp_min(1)          # [B, 1]
    return summed / denom


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Data
    loaders, tokenizer, label_maps = build_loaders_from_disk(
        args.data_dir, args.student_name, args.batch_size, args.num_workers
    )
    train_loader = loaders["train"]
    val_loader = loaders["validation"]
    num_labels = len(label_maps["id2label"])
    print(f"[INFO] Labels: {label_maps['id2label']} (num_labels={num_labels})")

    # Models
    teacher = AutoModelForSequenceClassification.from_pretrained(
        args.teacher_name, num_labels=num_labels,
        id2label=label_maps["id2label"], label2id=label_maps["label2id"]
    ).to(device)
    student = AutoModelForSequenceClassification.from_pretrained(
        args.student_name, num_labels=num_labels,
        id2label=label_maps["id2label"], label2id=label_maps["label2id"]
    ).to(device)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Connector to align feature dims (handles equal dims as identity)
    s_dim = student.config.hidden_size
    t_dim = teacher.config.hidden_size
    connector = Connector(in_dim=s_dim, out_dim=t_dim, use_ln=args.feat_ln).to(device)

    optimizer = AdamW(list(student.parameters()) + list(connector.parameters()),
                      lr=args.lr, weight_decay=args.weight_decay)

    # choose internal monitor if present
    Mon = _InternalMon if _InternalMon is not None else _NoOpMon

    # --- Training + Telemetry ---
    with Mon(outfile=f"{args.output_dir}/distill_power.csv") as mon:
        mon.log_once()  # baseline

        global_step = 0
        for epoch in range(1, args.epochs + 1):
            student.train()
            connector.train()
            run_tot = run_ce = run_kl = run_feat = 0.0
            seen = 0

            for batch in train_loader:
                # move to device and normalize dict
                batch = {k: v.to(device) for k, v in batch.items()}
                if "labels" not in batch:
                    raise KeyError(f"No 'labels' in batch. Keys: {list(batch.keys())}")
                labels = batch.pop("labels")

                with torch.no_grad():
                    t_out = teacher(**batch, output_hidden_states=True, return_dict=True)
                    t_logits = t_out.logits
                    t_hidden = t_out.hidden_states[args.t_layer]  # [B, L, D_t]

                s_out = student(**batch, output_hidden_states=True, return_dict=True)
                s_logits = s_out.logits
                s_hidden = s_out.hidden_states[args.s_layer]      # [B, L, D_s]

                # Feature pooling + connector
                attention_mask = batch["attention_mask"]
                t_feat = pool_tokens(t_hidden, attention_mask, args.feat_pool)  # [B, D_t]
                s_feat = pool_tokens(s_hidden, attention_mask, args.feat_pool)  # [B, D_s]
                s_feat_mapped = connector(s_feat)                                # [B, D_t]

                # Losses
                kd_total, ce, kl = kd_loss_logits(s_logits, t_logits, labels, T=args.kd_T, alpha=args.kd_alpha)
                f_loss = feature_loss(s_feat_mapped, t_feat, normalize=args.feat_normalize)
                total = kd_total + args.feat_beta * f_loss

                optimizer.zero_grad(set_to_none=True)
                total.backward()
                optimizer.step()

                # accumulate metrics
                bsz = labels.size(0)
                run_tot += total.item() * bsz
                run_ce  += ce.item()    * bsz
                run_kl  += kl.item()    * bsz
                run_feat+= f_loss.item()* bsz
                seen    += bsz
                global_step += 1

                # telemetry: log every N steps
                if hasattr(mon, "log_once") and args.telemetry_every > 0 and (global_step % args.telemetry_every == 0):
                    mon.log_once()

                if global_step % 50 == 0:
                    print(f"[Epoch {epoch} | Step {global_step}] "
                          f"Total {run_tot/seen:.4f} | CE {run_ce/seen:.4f} | KL {run_kl/seen:.4f} | Feat {run_feat/seen:.4f}")

            # epoch-end log
            if hasattr(mon, "log_once"):
                mon.log_once()
            print(f"[Epoch {epoch} DONE] "
                  f"Total {run_tot/seen:.4f} | CE {run_ce/seen:.4f} | KL {run_kl/seen:.4f} | Feat {run_feat/seen:.4f}")

            # Validation (student only)
            student.eval(); connector.eval()
            correct = count = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    if "labels" not in batch:
                        raise KeyError(f"No 'labels' in val batch. Keys: {list(batch.keys())}")
                    labels = batch.pop("labels")
                    logits = student(**batch).logits
                    pred = logits.argmax(dim=-1)
                    correct += (pred == labels).sum().item()
                    count += labels.size(0)
            if count:
                print(f"[Validation] Acc: {correct / count:.4f}")
            if hasattr(mon, "log_once"):
                mon.log_once()

    print("[INFO] Training completed.")


if __name__ == "__main__":
    main()
