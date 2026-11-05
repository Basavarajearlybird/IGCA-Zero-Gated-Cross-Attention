"""Training / evaluation pipeline shared by ``falcon_local.py`` and the Kaggle notebook. 

* Flickr8k image-captioning data (``jxie/flickr8k``: 6,000 train / 1,000 validation /
  1,000 test images). Training uses the train split; held-out loss uses a disjoint split
  (validation by default), so the held-out set can never be empty or overlap training.
* Optimiser construction with separate gate / weight learning rates.
* A run logger that writes a per-step loss log, an evaluation log and a JSON manifest
  (model actually loaded, insertion indices, trainable parameters, data sizes, epochs,
  peak memory, seconds per step), so every reported number can be traced to a file.
"""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import torch

from igca import IGCABridge

FLICKR8K_ID = "jxie/flickr8k"
FLICKR8K_SPLITS = {"train": 6000, "validation": 1000, "test": 1000}
PROMPT = "Image caption:"


# --------------------------------------------------------------------------- data
def load_flickr8k(train_rows: Optional[int] = None, heldout_split: str = "validation",
                  heldout_n: int = 200, caption_col: str = "caption_0"):
    """Return (train, heldout) datasets from disjoint Flickr8k splits."""
    if heldout_split == "train":
        raise ValueError("held-out data must come from a split disjoint from training")
    from datasets import load_dataset

    train = load_dataset(FLICKR8K_ID, split="train")
    if train_rows:
        train = train.select(range(min(train_rows, len(train))))
    held = load_dataset(FLICKR8K_ID, split=heldout_split)
    held = held.select(range(min(heldout_n, len(held))))
    if len(train) == 0 or len(held) == 0:
        raise ValueError(f"empty split: train={len(train)} heldout={len(held)}")
    for ds in (train, held):
        if caption_col not in ds.column_names:
            raise KeyError(f"{caption_col!r} not in {ds.column_names}")
    return train, held


class CaptionEncoder:
    """Prompt + caption tokenisation; loss is computed on caption tokens only."""

    def __init__(self, tokenizer, seqlen: int = 48, prompt: str = PROMPT,
                 caption_col: str = "caption_0", image_col: str = "image"):
        self.tok = tokenizer
        self.seqlen = seqlen
        self.caption_col = caption_col
        self.image_col = image_col
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.prompt_ids = self.tok(prompt, add_special_tokens=True).input_ids

    def encode(self, caption: str):
        cap = self.tok(" " + caption, add_special_tokens=False).input_ids
        ids = (self.prompt_ids + cap)[: self.seqlen]
        labels = ([-100] * len(self.prompt_ids) + cap)[: self.seqlen]
        return ids, labels

    def collate(self, examples, image_processor):
        images = [ex[self.image_col].convert("RGB") for ex in examples]
        pixel_values = image_processor(images=images, return_tensors="pt").pixel_values
        enc = [self.encode(ex[self.caption_col]) for ex in examples]
        maxlen = max(len(i) for i, _ in enc)
        pad = self.tok.pad_token_id
        ids, attn, labels = [], [], []
        for i, l in enc:
            n = maxlen - len(i)
            ids.append(i + [pad] * n)
            attn.append([1] * len(i) + [0] * n)
            labels.append(l + [-100] * n)
        return {"pixel_values": pixel_values, "input_ids": torch.tensor(ids),
                "attention_mask": torch.tensor(attn), "labels": torch.tensor(labels)}


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def embed_images(vision_encoder, pixel_values: torch.Tensor) -> torch.Tensor:
    dev = next(vision_encoder.parameters()).device
    dtype = next(vision_encoder.parameters()).dtype
    vision = getattr(vision_encoder, "vision_model", vision_encoder)
    return vision(pixel_values=pixel_values.to(dev, dtype)).last_hidden_state


@torch.no_grad()
def heldout_loss(llm, bridge: IGCABridge, heldout, enc: CaptionEncoder, vision_encoder,
                 image_processor, use_gate: bool) -> float:
    """Mean per-caption loss over the held-out set, gate on (IGCA) or off (frozen LLM)."""
    was_training = llm.training
    llm.eval()
    dev = next(llm.parameters()).device
    total, k = 0.0, 0
    for ex in heldout:
        ids, labels = enc.encode(ex[enc.caption_col])
        px = image_processor(images=ex[enc.image_col].convert("RGB"), return_tensors="pt").pixel_values
        bridge.set_modality(embed_images(vision_encoder, px))
        input_ids = torch.tensor([ids], device=dev)
        lab = torch.tensor([labels], device=dev)
        if use_gate:
            out = llm(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=lab)
        else:
            with bridge.disabled():
                out = llm(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=lab)
        total += float(out.loss.item())
        k += 1
    if was_training:
        llm.train()
    if k == 0:
        raise ValueError("held-out set is empty")
    return total / k


# --------------------------------------------------------------------------- optimisation
def build_optimizer(bridge: IGCABridge, lr_gate: float = 1e-3, lr_weight: float = 5e-5,
                    objective: str = "adamw", weight_decay: float = 0.01, paged_8bit: bool = True):
    """Two parameter groups (gate scalars get a higher LR).

    objective="adamw": decoupled weight decay on adapter weights (configuration of the
                       historical notebook).
    objective="alg1" : no decoupled decay; add ``mu * ||theta_IGCA||^2`` to the loss,
                       exactly as written in Algorithm 1.
    """
    if objective not in {"adamw", "alg1"}:
        raise ValueError("objective must be 'adamw' or 'alg1'")
    wd = weight_decay if objective == "adamw" else 0.0
    groups = []
    gates = bridge.gate_parameters()
    if gates:
        groups.append({"params": gates, "lr": lr_gate, "weight_decay": 0.0})
    groups.append({"params": bridge.weight_parameters(), "lr": lr_weight, "weight_decay": wd})
    if paged_8bit and torch.cuda.is_available():
        try:
            import bitsandbytes as bnb
            return bnb.optim.PagedAdamW8bit(groups), "PagedAdamW8bit"
        except Exception as exc:  # pragma: no cover - environment dependent
            print("PagedAdamW8bit unavailable -> torch AdamW:", repr(exc)[:120])
    return torch.optim.AdamW(groups), "AdamW"


def train_step(llm, bridge: IGCABridge, batch: dict, e_perp: torch.Tensor, optimizer,
               objective: str = "adamw", l2_mu: float = 0.0) -> float:
    dev = next(llm.parameters()).device
    bridge.set_modality(e_perp)
    out = llm(input_ids=batch["input_ids"].to(dev), attention_mask=batch["attention_mask"].to(dev),
              labels=batch["labels"].to(dev))
    loss = out.loss
    if objective == "alg1" and l2_mu > 0:
        loss = loss + l2_mu * bridge.l2_penalty().to(loss.device)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(out.loss.item())


# --------------------------------------------------------------------------- logging
@dataclass
class RunLogger:
    out_dir: str

    def __post_init__(self):
        os.makedirs(self.out_dir, exist_ok=True)
        self.t0 = time.time()
        self._loss_f = open(os.path.join(self.out_dir, "loss_log.csv"), "w", newline="")
        self._loss_w = csv.writer(self._loss_f)
        self._loss_w.writerow(["step", "train_loss", "mean_gate"])
        self._eval_f = open(os.path.join(self.out_dir, "eval_log.csv"), "w", newline="")
        self._eval_w = csv.writer(self._eval_f)
        self._eval_w.writerow(["step", "heldout_gate_on", "heldout_gate_off", "mean_gate"])

    def step(self, step: int, loss: float, gate: float):
        self._loss_w.writerow([step, f"{loss:.6f}", f"{gate:.6f}"])
        if step % 100 == 0:
            self._loss_f.flush()

    def eval(self, step: int, on: float, off: float, gate: float):
        self._eval_w.writerow([step, f"{on:.6f}", f"{off:.6f}", f"{gate:.6f}"])
        self._eval_f.flush()

    def manifest(self, data: dict, name: str = "run_manifest.json"):
        data = dict(data)
        data["wall_clock_s"] = round(time.time() - self.t0, 1)
        if torch.cuda.is_available():
            data["peak_memory_gb"] = {f"cuda:{i}": round(torch.cuda.max_memory_allocated(i) / 1e9, 3)
                                      for i in range(torch.cuda.device_count())}
        path = os.path.join(self.out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def close(self):
        self._loss_f.close()
        self._eval_f.close()


def model_slug(model_id: str) -> str:
    return model_id.split("/")[-1].replace(".", "-")


def decision(zero_gate_diff: float, mean_gate: float, delta: float, noise_floor: float = 0.02) -> str:
    """Pre-registered reading of a run (same rule as the original notebook)."""
    if zero_gate_diff != 0.0:
        return "ZERO-GATE FAILED: fix wiring; nothing downstream is trustworthy."
    if mean_gate <= 1e-3:
        return "VACUOUS: gate ~closed at best checkpoint; the delta is noise."
    if delta > noise_floor:
        return f"GROUNDED: gate-on beats gate-off by {delta:+.4f} nats with an open gate ({mean_gate:.4f})."
    if delta > 0:
        return f"INCONCLUSIVE: delta {delta:+.4f} is below the {noise_floor} noise floor."
    return f"NEGATIVE: gate-on is worse than gate-off ({delta:+.4f})."
