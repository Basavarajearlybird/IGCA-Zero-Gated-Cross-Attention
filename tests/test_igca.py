"""CPU test suite: runs in about a minute with no downloads (random-initialised backbones).

    pytest -q
"""
import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import igca  # noqa: E402
import igca_pipeline as pipe  # noqa: E402
import falcon_local  # noqa: E402


# --------------------------------------------------------------------------- accounting
@pytest.mark.parametrize("hidden,modality,heads", [(8192, 1024, 8), (4544, 1024, 8), (256, 64, 8), (64, 64, 4)])
@pytest.mark.parametrize("gated", [True, False])
def test_param_formula_matches_pytorch(hidden, modality, heads, gated):
    with torch.device("meta"):                      # no memory is allocated
        layer = igca.IGCALayer(hidden, modality, n_heads=heads, gated=gated)
    counted = sum(p.numel() for p in layer.parameters())
    assert counted == igca.params_per_module(hidden, modality, gated)


def test_falcon40b_accounting():
    rep = igca.param_report("tiiuae/falcon-40b", *igca.KNOWN_BACKBONES["tiiuae/falcon-40b"])
    assert rep.indices == [3, 7, 11, 15, 19, 23]
    assert rep.per_module == 151_027_713
    assert rep.total == 906_166_278


def test_insertion_indices():
    assert igca.insertion_indices(60) == [3, 7, 11, 15, 19, 23]
    assert igca.insertion_indices(8) == [3, 7]
    assert igca.insertion_indices(60, stride=4, max_modules=3) == [3, 7, 11]
    with pytest.raises(ValueError):
        igca.insertion_indices(10, stride=0)


# --------------------------------------------------------------------------- backbones
def _tiny(kind):
    torch.manual_seed(0)
    if kind == "falcon":
        model, _, _ = falcon_local.tiny_random_falcon()
        return model
    if kind == "llama":
        from transformers import LlamaConfig, LlamaForCausalLM
        return LlamaForCausalLM(LlamaConfig(vocab_size=512, hidden_size=128, intermediate_size=256,
                                            num_hidden_layers=8, num_attention_heads=8))
    if kind == "bloom":
        from transformers import BloomConfig, BloomForCausalLM
        return BloomForCausalLM(BloomConfig(vocab_size=512, hidden_size=128, n_layer=8, n_head=8))
    raise ValueError(kind)


@pytest.mark.parametrize("kind,path", [("falcon", "transformer.h"), ("llama", "model.layers"),
                                       ("bloom", "transformer.h")])
def test_zero_gate_is_exact_no_op(kind, path):
    model = _tiny(kind).eval()
    igca.freeze(model)
    bridge = igca.IGCABridge(model, modality_dim=32)
    assert bridge.layer_path == path and bridge.indices == [3, 7]
    ids = torch.randint(1, 500, (2, 10))
    diff = igca.zero_gate_max_diff(model, bridge, {"input_ids": ids}, torch.randn(2, 5, 32))
    assert diff == 0.0                                # bit-exact, not merely close


def test_ungated_adapter_perturbs_at_init():
    model = _tiny("falcon").eval()
    igca.freeze(model)
    bridge = igca.IGCABridge(model, modality_dim=32, gated=False)
    ids = torch.randint(1, 500, (1, 10))
    assert igca.zero_gate_max_diff(model, bridge, {"input_ids": ids}, torch.randn(1, 5, 32)) > 1e-3


def test_only_adapter_receives_gradients():
    model = _tiny("falcon").train()
    igca.freeze(model)
    bridge = igca.IGCABridge(model, modality_dim=32)
    task = falcon_local.SyntheticGrounding(model.config.vocab_size, 32, seqlen=16)
    batch, e = task.batch(2, seed=0)
    bridge.set_modality(e)
    model(**batch).loss.backward()
    assert all(p.grad is None for p in model.parameters())          # backbone untouched
    assert all(m.alpha.grad is not None and m.alpha.grad.abs() > 0 for m in bridge.modules)
    # at alpha = 0 the attention weights receive exactly zero gradient (gate closes the path)
    assert all(p.grad is None or p.grad.abs().max() == 0 for p in bridge.weight_parameters())


def test_disable_and_remove_restore_backbone():
    model = _tiny("falcon").eval()
    igca.freeze(model)
    ids = torch.randint(1, 500, (1, 8))
    ref = model(input_ids=ids).logits
    bridge = igca.IGCABridge(model, modality_dim=32, gated=False)
    bridge.set_modality(torch.randn(1, 5, 32))
    with bridge.disabled():
        assert torch.equal(model(input_ids=ids).logits, ref)
    bridge.remove()
    assert torch.equal(model(input_ids=ids).logits, ref)


def test_alg1_l2_penalty_matches_definition():
    model = _tiny("falcon")
    bridge = igca.IGCABridge(model, modality_dim=32)
    manual = sum(p.detach().float().pow(2).sum() for p in bridge.parameters())
    assert torch.allclose(bridge.l2_penalty().detach(), manual)


# --------------------------------------------------------------------------- pipeline
def test_heldout_split_must_be_disjoint():
    with pytest.raises(ValueError):
        pipe.load_flickr8k(heldout_split="train")


def test_decision_rule():
    assert pipe.decision(0.0, 0.07, 2.1).startswith("GROUNDED")
    assert pipe.decision(0.0, 0.0, 2.1).startswith("VACUOUS")
    assert pipe.decision(1e-6, 0.07, 2.1).startswith("ZERO-GATE FAILED")
    assert pipe.decision(0.0, 0.07, 0.01).startswith("INCONCLUSIVE")


def test_end_to_end_grounding_run(tmp_path):
    out = tmp_path / "run"
    code = falcon_local.main(["--tiny-random", "--steps", "150", "--eval-every", "75",
                              "--out-dir", str(out)])
    assert code == 0
    m = json.loads((out / "run_manifest.json").read_text())
    assert m["zero_gate_max_diff"] == 0.0
    assert m["step0_shift_nats"] == 0.0
    assert m["trainable_params"] == igca.params_per_module(256, 64) * 2
    assert m["mean_gate_at_best"] > 1e-3                        # the gate opened
    assert m["best_heldout_gate_on"] < m["heldout_gate_off"] - 0.5   # information flows through IGCA
    assert (out / "loss_log.csv").exists() and (out / "eval_log.csv").exists()


def _offline_caption_stack():
    """Tiny tokenizer + random CLIP vision tower + in-memory 'Flickr8k' rows (no downloads)."""
    import numpy as np
    from PIL import Image
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel, PreTrainedTokenizerFast
    words = ["[PAD]", "[UNK]", "</s>", "Image", "caption:", "a", "dog", "runs", "on", "grass", "cat", "sits"]
    tk = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", eos_token="</s>")
    vision = CLIPVisionModel(CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                              num_attention_heads=4, image_size=32, patch_size=8)).eval()
    proc = CLIPImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32})
    rng = np.random.default_rng(0)
    rows = [{"image": Image.fromarray(rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)),
             "caption_0": c} for c in ["a dog runs on grass", "a cat sits", "a dog sits", "a cat runs"]]
    return tok, vision, proc, rows


def test_caption_pipeline_offline():
    tok, vision, proc, rows = _offline_caption_stack()
    model = _tiny("falcon")
    igca.freeze(model)
    bridge = igca.IGCABridge(model, modality_dim=32)
    enc = pipe.CaptionEncoder(tok, seqlen=48)
    ids, labels = enc.encode("a dog runs")
    assert labels[: len(enc.prompt_ids)] == [-100] * len(enc.prompt_ids)      # prompt is not scored
    batch = enc.collate(rows[:2], proc)
    assert batch["input_ids"].shape == batch["labels"].shape
    off = pipe.heldout_loss(model, bridge, rows, enc, vision, proc, use_gate=False)
    on = pipe.heldout_loss(model, bridge, rows, enc, vision, proc, use_gate=True)
    assert off == on                                   # alpha = 0: identical held-out loss
    opt, _ = pipe.build_optimizer(bridge, 1e-2, 1e-3, paged_8bit=False)
    loss = pipe.train_step(model, bridge, batch, pipe.embed_images(vision, batch["pixel_values"]), opt)
    assert loss > 0 and bridge.mean_gate() > 0         # one update opens the gate

