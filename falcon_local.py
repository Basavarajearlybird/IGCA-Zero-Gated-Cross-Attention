"""IGCA on Falcon (or any HF causal LM): one script, CPU to multi-GPU.

Modes
-----
--tiny-random      Random-initialised 8-layer Falcon built from a config (no download).
                   Verifies layer discovery, hook placement, the exact zero-gate identity,
                   gradient isolation and gate opening in seconds on a laptop CPU.
--data synthetic   Real checkpoint, synthetic tokens + modality embeddings (wiring check).
--data flickr8k    Real checkpoint + CLIP ViT-L/14 + Flickr8k captioning, held-out loss on
                   the disjoint validation split (the experiment of Section 5).

Every run writes <out-dir>/run_manifest.json, loss_log.csv and eval_log.csv.
The model that is actually loaded is the one recorded; nothing falls back silently.

Examples
--------
python falcon_local.py --tiny-random --steps 20
python falcon_local.py --model tiiuae/falcon-7b  --data flickr8k --steps 2000
python falcon_local.py --model tiiuae/falcon-40b --data flickr8k --steps 59500 --variant gated
python falcon_local.py --model tiiuae/falcon-40b --data flickr8k --steps 59500 --variant ungated
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

import igca
import igca_pipeline as pipe


# --------------------------------------------------------------------------- loading
def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def tiny_random_falcon(seed: int = 0):
    """8-layer, 256-wide Falcon with random weights: exercises the real Falcon code path."""
    from transformers import FalconConfig, FalconForCausalLM
    torch.manual_seed(seed)
    cfg = FalconConfig(vocab_size=1024, hidden_size=256, num_hidden_layers=8,
                       num_attention_heads=8, new_decoder_architecture=False,
                       parallel_attn=True, bias=False, alibi=False)
    return FalconForCausalLM(cfg), None, "tiny-random-falcon"


def load_backbone(model_id: str, device: str, use_4bit: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {}
    if use_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16)
        kw["device_map"] = "auto"
        kw["torch_dtype"] = torch.float16
    else:
        kw["torch_dtype"] = torch.float32
    llm = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    if not use_4bit:
        llm = llm.to(device)
    return llm, tok, model_id


class SyntheticGrounding:
    """Toy cross-modal task with a known answer.

    Each example belongs to one of ``n_concepts`` concepts. The modality embedding is the
    concept's fixed prototype plus noise; the text is a random prefix followed by the
    concept's fixed 4-token "caption". The caption is unpredictable from the text alone,
    so held-out loss can only fall if information flows through the IGCA bridge.
    """

    def __init__(self, vocab: int, modality_dim: int, seqlen: int, n_concepts: int = 16,
                 caption_len: int = 4, n_tokens: int = 16, noise: float = 0.1, seed: int = 777):
        g = torch.Generator().manual_seed(seed)
        self.vocab, self.seqlen, self.noise = vocab, seqlen, noise
        self.prototypes = torch.randn(n_concepts, n_tokens, modality_dim, generator=g)
        self.captions = torch.randint(1, vocab, (n_concepts, caption_len), generator=g)
        self.n_concepts, self.caption_len = n_concepts, caption_len

    def batch(self, batch: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        c = torch.randint(0, self.n_concepts, (batch,), generator=g)
        prefix = torch.randint(1, self.vocab, (batch, self.seqlen - self.caption_len), generator=g)
        ids = torch.cat([prefix, self.captions[c]], dim=1)
        labels = ids.clone()
        labels[:, : self.seqlen - self.caption_len] = -100      # score the caption only
        e = self.prototypes[c] + self.noise * torch.randn(self.prototypes[c].shape, generator=g)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}, e


def synthetic_probe(vocab: int, modality_dim: int, seed: int = 1234):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, vocab, (1, 12), generator=g), torch.randn(1, 16, modality_dim, generator=g)


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="HF model id (ignored with --tiny-random)")
    ap.add_argument("--tiny-random", action="store_true")
    ap.add_argument("--data", choices=["synthetic", "flickr8k"], default="synthetic")
    ap.add_argument("--vision-encoder", default="openai/clip-vit-large-patch14")
    ap.add_argument("--variant", choices=["gated", "ungated"], default="gated",
                    help="gated = IGCA (Eq. 2); ungated = standard adapter baseline (Eq. 1)")
    ap.add_argument("--steps", type=int, default=None, help="optimizer updates")
    ap.add_argument("--train-rows", type=int, default=None, help="cap on training captions")
    ap.add_argument("--heldout-split", default="validation")
    ap.add_argument("--heldout-n", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seqlen", type=int, default=48)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--modules", type=int, default=6)
    ap.add_argument("--nheads", type=int, default=8)
    ap.add_argument("--lr-gate", type=float, default=None, help="default 1e-3 (1e-2 tiny)")
    ap.add_argument("--lr-weight", type=float, default=None, help="default 5e-5 (1e-3 tiny)")
    ap.add_argument("--objective", choices=["adamw", "alg1"], default="adamw")
    ap.add_argument("--l2-mu", type=float, default=0.0, help="mu for --objective alg1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    device = pick_device()
    use_4bit = device == "cuda" and not a.tiny_random
    if a.steps is None:
        a.steps = 300 if (a.tiny_random or device == "cpu") else 2000
    if a.lr_gate is None:
        a.lr_gate = 1e-2 if a.tiny_random else 1e-3
    if a.lr_weight is None:
        a.lr_weight = 1e-3 if a.tiny_random else 5e-5

    print("=" * 72)
    print(f"IGCA v{igca.__version__} | device={device} | torch {torch.__version__}")
    print("=" * 72)

    # ---- backbone (no silent fallback) ------------------------------------------
    if a.tiny_random:
        llm, tok, model_id = tiny_random_falcon(a.seed)
        llm = llm.to(device)
        if a.data == "flickr8k":
            ap.error("--tiny-random has no tokenizer; use --data synthetic")
    else:
        if a.model is None:
            ap.error("--model is required unless --tiny-random is given")
        try:
            llm, tok, model_id = load_backbone(a.model, device, use_4bit)
        except Exception as exc:
            print(f"\nCould not load {a.model}: {type(exc).__name__}: {str(exc)[:200]}")
            print("No fallback is attempted: choose a model that fits this machine explicitly.")
            return 3
    frozen = igca.freeze(llm)
    if hasattr(llm, "gradient_checkpointing_enable") and device == "cuda":
        llm.gradient_checkpointing_enable()
        llm.enable_input_require_grads()
        llm.config.use_cache = False
    llm.train()

    # ---- modality encoder ----------------------------------------------------------
    vision = image_proc = None
    modality_dim = igca.CLIP_VIT_L14_DIM
    if a.data == "flickr8k":
        from transformers import AutoImageProcessor, AutoModel
        vdtype = torch.float16 if device == "cuda" else torch.float32
        vision = AutoModel.from_pretrained(a.vision_encoder, torch_dtype=vdtype).to(
            "cuda:0" if device == "cuda" else device).eval()
        igca.freeze(vision)
        image_proc = AutoImageProcessor.from_pretrained(a.vision_encoder)
        modality_dim = vision.config.vision_config.hidden_size
    elif a.tiny_random:
        modality_dim = 64

    # ---- bridge --------------------------------------------------------------------
    dtype = torch.float16 if use_4bit else torch.float32
    bridge = igca.IGCABridge(llm, modality_dim, stride=a.stride, max_modules=a.modules,
                             n_heads=a.nheads, gated=a.variant == "gated", dtype=dtype)
    n_train = bridge.num_trainable()
    assert n_train == bridge.expected_trainable(), "parameter count disagrees with the analytic formula"
    print(f"model            : {model_id}")
    print(f"decoder          : {bridge.layer_path} ({bridge.num_layers} layers, hidden {bridge.hidden})")
    print(f"IGCA layers      : {bridge.indices} | variant={a.variant}")
    print(f"trainable params : {n_train:,} (~{n_train / 1e9:.3f}B) | frozen: {frozen:,}")

    # ---- exact zero-gate identity --------------------------------------------------
    vocab = llm.config.vocab_size
    probe_ids, e_probe = synthetic_probe(vocab, modality_dim)
    dev = next(llm.parameters()).device
    # Max |logits(bridge on) - logits(bridge off)| before training. IGCA: exactly 0.
    # The un-gated adapter (Eq. 1) is already perturbing the frozen model here.
    step0_diff = igca.zero_gate_max_diff(llm, bridge, {"input_ids": probe_ids.to(dev)}, e_probe.to(dev))
    zero_diff = step0_diff if a.variant == "gated" else None
    print(f"step-0 max |logit diff| (bridge on vs off): {step0_diff:.2e}"
          + (f" ({'EXACT no-op' if step0_diff == 0.0 else 'FAIL'})" if a.variant == "gated"
             else " (un-gated adapter perturbs the frozen model at init)"))
    if a.variant == "gated" and step0_diff != 0.0:
        return 2

    # ---- data ----------------------------------------------------------------------
    data_info = {"dataset": a.data}
    if a.data == "flickr8k":
        train, held = pipe.load_flickr8k(a.train_rows, a.heldout_split, a.heldout_n)
        enc = pipe.CaptionEncoder(tok, a.seqlen)
        loader = torch.utils.data.DataLoader(
            train, batch_size=a.batch, shuffle=True,
            collate_fn=lambda ex: enc.collate(ex, image_proc))
        it = pipe.infinite(loader)
        data_info.update(train_rows=len(train), heldout_split=a.heldout_split,
                         heldout_rows=len(held),
                         epochs=round(a.steps * a.batch / len(train), 2))
        print(f"data             : Flickr8k train {len(train)} | held-out "
              f"{a.heldout_split} {len(held)} | {data_info['epochs']} epochs")

        def next_batch(step):
            b = next(it)
            return b, pipe.embed_images(vision, b["pixel_values"])

        def evaluate(gate_on):
            return pipe.heldout_loss(llm, bridge, held, enc, vision, image_proc, gate_on)
    else:
        task = SyntheticGrounding(vocab, modality_dim, a.seqlen)
        data_info.update(task="synthetic grounding (16 concepts, 4-token captions)",
                         heldout_rows=32)

        def next_batch(step):
            return task.batch(a.batch, seed=a.seed * 1_000_003 + step)

        held_batches = [task.batch(1, seed=10_000_000 + i) for i in range(32)]

        @torch.no_grad()
        def evaluate(gate_on):
            llm.eval()
            losses = []
            for b, e in held_batches:
                bridge.set_modality(e.to(dev))
                kw = {k: v.to(dev) for k, v in b.items()}
                if gate_on:
                    losses.append(float(llm(**kw).loss))
                else:
                    with bridge.disabled():
                        losses.append(float(llm(**kw).loss))
            llm.train()
            return sum(losses) / len(losses)

    # ---- train ---------------------------------------------------------------------
    optimizer, opt_name = pipe.build_optimizer(bridge, a.lr_gate, a.lr_weight, a.objective,
                                               paged_8bit=use_4bit)
    slug = pipe.model_slug(model_id)
    out_dir = a.out_dir or os.path.join("runs", f"{slug}_{a.variant}_{time.strftime('%Y%m%d-%H%M%S')}")
    log = pipe.RunLogger(out_dir)
    off = evaluate(False)
    on0 = evaluate(True)                      # bridge attached, before any update
    print(f"held-out gate-OFF (frozen backbone): {off:.4f}")
    print(f"held-out step 0 with bridge active : {on0:.4f} "
          f"(shift {on0 - off:+.4f} nats)")
    log.eval(0, on0, off, bridge.mean_gate())

    best = {"loss": float("inf"), "step": 0, "gate": 0.0}
    first = last = None
    t0 = time.time()
    eval_every = min(a.eval_every, a.steps)
    for step in range(1, a.steps + 1):
        batch, e = next_batch(step)
        loss = pipe.train_step(llm, bridge, batch, e.to(dev), optimizer, a.objective, a.l2_mu)
        first = loss if first is None else first
        last = loss
        log.step(step, loss, bridge.mean_gate())
        if step % eval_every == 0 or step == a.steps:
            on = evaluate(True)
            g = bridge.mean_gate()
            log.eval(step, on, off, g)
            print(f"step {step:>6} | train {loss:.4f} | held-out on {on:.4f} (off {off:.4f}) | gate {g:.4f}")
            if on < best["loss"]:
                best = {"loss": on, "step": step, "gate": g}
                torch.save({"igca_state": bridge.modules.state_dict(), "step": step,
                            "model_id": model_id, "indices": bridge.indices, "variant": a.variant},
                           os.path.join(out_dir, f"igca_best_{slug}.pt"))
    sec_per_step = (time.time() - t0) / a.steps

    delta = off - best["loss"]
    verdict = pipe.decision(zero_diff if zero_diff is not None else 0.0, best["gate"], delta) \
        if a.variant == "gated" else f"UNGATED BASELINE: best held-out {best['loss']:.4f} (off {off:.4f})"
    manifest = igca.manifest_base(
        model_id=model_id, device=device, quantization="nf4-4bit" if use_4bit else "none",
        variant=a.variant, layer_path=bridge.layer_path, num_layers=bridge.num_layers,
        hidden_size=bridge.hidden, modality_dim=modality_dim, igca_indices=bridge.indices,
        n_heads=a.nheads, trainable_params=n_train, frozen_params=frozen,
        zero_gate_max_diff=zero_diff, step0_max_logit_diff=step0_diff, optimizer=opt_name, objective=a.objective, l2_mu=a.l2_mu,
        lr_gate=a.lr_gate, lr_weight=a.lr_weight, steps=a.steps, batch=a.batch, seqlen=a.seqlen,
        seed=a.seed, data=data_info, train_loss_first=first, train_loss_last=last,
        heldout_gate_off=off, heldout_step0_bridge_active=on0, step0_shift_nats=on0 - off,
        best_heldout_gate_on=best["loss"], best_step=best["step"],
        mean_gate_at_best=best["gate"], heldout_delta_nats=delta,
        heldout_delta_pct=100 * delta / off if off else None,
        sec_per_step=round(sec_per_step, 4), decision=verdict)
    path = log.manifest(manifest)
    log.close()
    print("-" * 72)
    print(f"decision : {verdict}")
    print(f"manifest : {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
