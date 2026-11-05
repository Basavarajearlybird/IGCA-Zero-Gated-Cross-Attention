"""IGCA: zero-gated cross-attention bridge for frozen decoder language models.

Reference implementation for "Zero-Gated Cross-Attention" (arXiv:2611.01116).

At each selected decoder layer l the bridge computes

    h'_l = h_l + tanh(alpha_l) * MHA(h_l W_Q, E W_K, E W_V),   alpha_l initialised to 0,

so the augmented model is *exactly* the frozen backbone at initialisation
(tanh(0) = 0). The tanh-gated, zero-initialised cross-attention principle follows
Flamingo (Alayrac et al., 2022); this module isolates it as a standalone bridge for
frozen, optionally 4-bit-quantised decoders.

This file is the single source of truth used by ``falcon_local.py``, the Kaggle
notebook and the test suite.

Command-line parameter accounting (no download needed for known backbones):

    python igca.py --count tiiuae/falcon-40b
    python igca.py --hidden 8192 --modality 1024 --layers 60
"""
from __future__ import annotations

import argparse
import contextlib
import json
from dataclasses import dataclass, asdict
from typing import Iterable, Optional

import torch
import torch.nn as nn

__version__ = "1.1.0"

# Published configurations (hidden size, decoder layers) used for offline parameter
# accounting. Anything else is read from the Hugging Face config with --count <model-id>.
KNOWN_BACKBONES = {
    "tiiuae/falcon-40b": (8192, 60),
    "tiiuae/falcon-7b": (4544, 32),
    "tiiuae/falcon-rw-1b": (2048, 24),
    "bigscience/bloom-7b1": (4096, 30),
}
CLIP_VIT_L14_DIM = 1024          # openai/clip-vit-large-patch14 vision hidden size
ARXIV_V1_REPORTED_PARAMS = 40_404_961_288   # see ERRATA.md


# --------------------------------------------------------------------------- layer
class IGCALayer(nn.Module):
    """One gated cross-attention bridge.

    Query = decoder hidden state, key/value = modality embeddings. With ``gated=True``
    (IGCA) the update is scaled by ``tanh(alpha)`` with ``alpha`` initialised to zero.
    With ``gated=False`` the module is the standard un-gated adapter of Eq. (1), used as
    the ablation baseline.
    """

    def __init__(self, hidden_dim: int, modality_dim: int, n_heads: int = 8, gated: bool = True):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by n_heads={n_heads}")
        self.gated = gated
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads,
            kdim=modality_dim, vdim=modality_dim, batch_first=True,
        )
        if gated:
            self.alpha = nn.Parameter(torch.zeros(1))      # the zero-initialised gate
        else:
            self.register_parameter("alpha", None)

    def gate_value(self) -> torch.Tensor:
        if not self.gated:
            return torch.ones(())
        return torch.tanh(self.alpha).reshape(())

    def forward(self, h_l: torch.Tensor, e_perp: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.mha(h_l, e_perp, e_perp, need_weights=False)
        if self.gated:
            return h_l + torch.tanh(self.alpha) * attn_out
        return h_l + attn_out


# --------------------------------------------------------------------------- accounting
def params_per_module(hidden: int, modality: int, gated: bool = True) -> int:
    """Exact trainable-parameter count of one IGCALayer (nn.MultiheadAttention layout).

    W_Q: H*H, W_K: H*D, W_V: H*D, in-proj bias: 3H, W_O: H*H, out bias: H, gate: 1.
    """
    return 2 * hidden * hidden + 2 * hidden * modality + 4 * hidden + (1 if gated else 0)


def insertion_indices(num_layers: int, stride: int = 4, max_modules: int = 6) -> list[int]:
    """Stride policy: layers stride-1, 2*stride-1, ... capped at max_modules."""
    if stride < 1 or max_modules < 1:
        raise ValueError("stride and max_modules must be >= 1")
    return list(range(stride - 1, num_layers, stride))[:max_modules]


@dataclass
class ParamReport:
    backbone: str
    hidden: int
    modality: int
    num_layers: int
    indices: list
    per_module: int
    total: int
    gated: bool

    def as_text(self) -> str:
        lines = [
            f"backbone              : {self.backbone}",
            f"hidden / modality dim : {self.hidden} / {self.modality}",
            f"decoder layers        : {self.num_layers}",
            f"IGCA insertion layers : {self.indices} ({len(self.indices)} modules)",
            f"params per module     : {self.per_module:,}",
            f"total trainable       : {self.total:,}  (~{self.total / 1e9:.3f}B)",
        ]
        return "\n".join(lines)


def param_report(backbone: str, hidden: int, num_layers: int, modality: int = CLIP_VIT_L14_DIM,
                 stride: int = 4, max_modules: int = 6, gated: bool = True) -> ParamReport:
    idx = insertion_indices(num_layers, stride, max_modules)
    per = params_per_module(hidden, modality, gated)
    return ParamReport(backbone, hidden, modality, num_layers, idx, per, per * len(idx), gated)


# --------------------------------------------------------------------------- wiring
_LAYER_PATHS = [
    ("transformer", "h"),            # Falcon, BLOOM, GPT-2
    ("model", "layers"),             # LLaMA / OpenLLaMA / Mistral / Qwen
    ("model", "decoder", "layers"),  # OPT
    ("gpt_neox", "layers"),          # GPT-NeoX / Pythia
    ("transformer", "blocks"),       # MPT
]


def find_decoder_layers(model: nn.Module):
    """Return (ModuleList of decoder layers, dotted path) for common HF causal LMs."""
    for path in _LAYER_PATHS:
        obj, ok = model, True
        for attr in path:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and isinstance(obj, nn.ModuleList) and len(obj) > 0:
            return obj, ".".join(path)
    raise RuntimeError("Could not auto-detect decoder layers; inspect model.named_modules().")


def module_device(module: nn.Module) -> torch.device:
    for p in module.parameters():
        return p.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class IGCABridge:
    """Attach IGCA modules to a frozen decoder through forward hooks.

    Each module is placed on the same device as the layer it hooks, so the bridge works
    with ``device_map="auto"`` sharding. Call :meth:`set_modality` before every forward
    pass and :meth:`disabled` to obtain the gate-off (frozen backbone) reference.
    """

    def __init__(self, model: nn.Module, modality_dim: int, stride: int = 4, max_modules: int = 6,
                 n_heads: int = 8, gated: bool = True, dtype: Optional[torch.dtype] = None):
        layers, self.layer_path = find_decoder_layers(model)
        self.num_layers = len(layers)
        self.hidden = model.config.hidden_size
        self.modality_dim = modality_dim
        self.gated = gated
        self.indices = insertion_indices(self.num_layers, stride, max_modules)
        self.modules = nn.ModuleList()
        for idx in self.indices:
            m = IGCALayer(self.hidden, modality_dim, n_heads=n_heads, gated=gated)
            if dtype is not None:
                m.to(dtype)
            m.to(module_device(layers[idx]))
            self.modules.append(m)
        self._e_perp: Optional[torch.Tensor] = None
        self.enabled = True
        self.handles = [layers[idx].register_forward_hook(self._make_hook(m))
                        for m, idx in zip(self.modules, self.indices)]

    # -- hooks ------------------------------------------------------------------
    def _make_hook(self, layer: IGCALayer):
        def hook(_module, _inputs, output):
            if not self.enabled or self._e_perp is None:
                return output
            hs = output[0] if isinstance(output, tuple) else output
            e = self._e_perp.to(device=hs.device, dtype=hs.dtype)
            new = layer(hs, e)
            return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new
        return hook

    def set_modality(self, e_perp: Optional[torch.Tensor]) -> None:
        self._e_perp = e_perp

    @contextlib.contextmanager
    def disabled(self):
        prev = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = prev

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    # -- introspection ----------------------------------------------------------
    def parameters(self) -> Iterable[nn.Parameter]:
        return self.modules.parameters()

    def gate_parameters(self) -> list:
        return [m.alpha for m in self.modules if m.gated]

    def weight_parameters(self) -> list:
        return [p for n, p in self.modules.named_parameters() if not n.endswith("alpha")]

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.modules.parameters())

    def expected_trainable(self) -> int:
        return len(self.indices) * params_per_module(self.hidden, self.modality_dim, self.gated)

    def mean_gate(self) -> float:
        with torch.no_grad():
            if not self.gated:
                return 1.0
            return float(torch.stack([m.gate_value().abs().cpu().float() for m in self.modules]).mean())

    def l2_penalty(self) -> torch.Tensor:
        """sum ||theta_IGCA||^2, the explicit regulariser of Algorithm 1."""
        total = None
        for p in self.modules.parameters():
            term = p.float().pow(2).sum()
            total = term if total is None else total + term.to(total.device)
        return total


def freeze(model: nn.Module) -> int:
    """Freeze every backbone parameter; returns how many were frozen."""
    n = 0
    for p in model.parameters():
        p.requires_grad = False
        n += p.numel()
    return n


@torch.no_grad()
def zero_gate_max_diff(model: nn.Module, bridge: IGCABridge, model_inputs: dict,
                       e_perp: torch.Tensor) -> float:
    """Max |logits(gate-on, alpha=0) - logits(gate-off)|. Exactly 0.0 for a fresh IGCA."""
    was_training = model.training
    model.eval()
    bridge.set_modality(e_perp)
    with bridge.disabled():
        off = model(**model_inputs).logits.float()
    on = model(**model_inputs).logits.float()
    if was_training:
        model.train()
    return float((off - on).abs().max().item())


def manifest_base(**fields) -> dict:
    """Common header for run manifests written by the script and notebook."""
    import platform
    import transformers
    head = {
        "igca_version": __version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    head.update(fields)
    return head


# --------------------------------------------------------------------------- CLI
def _main() -> None:
    ap = argparse.ArgumentParser(description="IGCA trainable-parameter accounting")
    ap.add_argument("--count", metavar="MODEL_ID", help="HF model id (known ids work offline)")
    ap.add_argument("--hidden", type=int)
    ap.add_argument("--layers", type=int)
    ap.add_argument("--modality", type=int, default=CLIP_VIT_L14_DIM)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--modules", type=int, default=6)
    ap.add_argument("--ungated", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    name = a.count or "custom"
    hidden, layers = a.hidden, a.layers
    if a.count and (hidden is None or layers is None):
        if a.count in KNOWN_BACKBONES:
            hidden, layers = KNOWN_BACKBONES[a.count]
        else:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(a.count)
            hidden = cfg.hidden_size
            layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer")
    if hidden is None or layers is None:
        ap.error("give --count MODEL_ID or both --hidden and --layers")

    rep = param_report(name, hidden, layers, a.modality, a.stride, a.modules, not a.ungated)
    if a.json:
        print(json.dumps(asdict(rep), indent=2))
        return
    print(rep.as_text())
    if name == "tiiuae/falcon-40b" and not a.ungated and rep.total != ARXIV_V1_REPORTED_PARAMS:
        print(f"note                  : arXiv v1 printed {ARXIV_V1_REPORTED_PARAMS:,}; "
              "the architecture above is authoritative (see ERRATA.md).")


if __name__ == "__main__":
    _main()
