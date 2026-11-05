# Notes on arXiv:2611.01116v1

These notes reconcile the v1 preprint with implementation release v1.1 and will be folded into
the next arXiv version. The parameter count and the average held-out improvement are corrected
below; the IGCA mechanism and the exact zero-gate identity are unchanged.

The historical best-checkpoint results (4.3469 → 2.2201 nats, −48.9%) are retained as reported
in v1. The original checkpoint and raw logs were not archived, and the v1 notebook's held-out
slice was empty under the documented Flickr8k split, so these values cannot be regenerated from
the v1 notebook. v1.1 provides a corrected pipeline (disjoint held-out data, persistent logging);
new runs with it are new experiments, not a retrospective verification of the v1 figures.

| # | Item | v1 preprint | Corrected / clarified | How to check |
|:-:|---|---|---|---|
| 1 | Trainable adapter parameters (Falcon-40B, 6 modules) | 40,404,961,288 | **906,166,278** = 6 × 151,027,713 (H = 8192, D = 1024). v1 figure is a transcription error. | `python igca.py --count tiiuae/falcon-40b` |
| 2 | Average held-out improvement | 1.9075 nats | **2.0925 nats** (4.3423 − 2.2498; −48.2%). Best-checkpoint figures (2.1268 nats, −48.9%) are correct. | arithmetic |
| 3 | Training data | not named | Flickr8k (`jxie/flickr8k`, `caption_0`), CLIP ViT-L/14 image tokens, prompt `Image caption:`; 6,000 image–caption pairs (one caption per image). The v1 notebook's intended 30-caption held-out slice (`train[59500:59530]`) was empty. v1.1 evaluates on the disjoint validation split (default 200, up to 1,000). | `igca_pipeline.py` |
| 4 | Train loss 3.9403 → 0.0007 | reported without epoch count | 59,500 updates at batch 1 ≈ 9.9 passes over the 6,000 training pairs. The near-zero training loss is consistent with memorisation and is not evidence of generalisation. Because the v1 notebook's held-out slice was empty, historical checkpoint selection on valid held-out data cannot be established; in v1.1, checkpoint selection uses the disjoint validation split. | manifest field `data.epochs` |
| 5 | Related work | Flamingo not cited | The tanh-gated, zero-initialised cross-attention principle follows **Flamingo** (Alayrac et al., NeurIPS 2022). IGCA's contribution is isolating it as a stand-alone bridge on a frozen, 4-bit-quantised 40B decoder with bit-exact identity verification. | — |
| 6 | OpenFlamingo reference | NeurIPS 2023 | arXiv technical report (arXiv:2308.01390). | — |

## Implementation changes in v1.1

| v1 notebook | v1.1 |
|---|---|
| Held-out slice `train[59500:59530]` lies beyond Flickr8k's 6,000 training rows (empty set) | Disjoint validation split; empty or overlapping held-out sets raise an error |
| `N_TRAIN` and `MAX_STEPS` shared one value | Dataset size and optimiser updates are separate settings; epochs are logged |
| Automatic fallback to Falcon-7B / Falcon-RW-1B when resources were short, with `falcon40b` file names | No fallback: the run stops with a clear message, and the loaded model is written to the manifest and file names |
| Zero-gate tolerance `< 1e-3` | Bit-exact check (`== 0.0`) |
| Gate-off (frozen backbone) baseline only | Adds the un-gated adapter of Eq. (1) as an ablation (`--variant ungated`) |
| Decoupled AdamW weight decay only | Also the explicit `μ‖θ‖²` objective of Algorithm 1 (`--objective alg1 --l2-mu`) |
| No run record written | `run_manifest.json`, `loss_log.csv`, `eval_log.csv` for every run |

## Scope reminders (already stated in v1)

* Empirical results are for Falcon-40B only. Open_llama_30b and Bloom-7b1 are supported
  architectures (hook placement and exact identity are tested on LLaMA and BLOOM layouts),
  not evaluated backbones.
* The released code implements the image modality (CLIP ViT-L/14). Audio (Whisper) and video
  (VideoMAE v2) encoders in Algorithm 1 are design scope.
* Figure 2 of v1 is a schematic. The historical run's raw log and checkpoint were not archived;
  v1.1 writes both for every new run.
