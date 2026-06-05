# Gemma 4 12B-it — Encoder-Free Architecture (research notes)

Source of truth: `transformers.models.gemma4_unified` (v5.10.0.dev0) + the checkpoint
`google/gemma-4-12B-it` `config.json`. Everything below was verified against code, not marketing.

## 0. The single most important framing for the paper

The transformers library ships **two distinct Gemma 4 implementations**:

| module | model_type | class | vision/audio |
|---|---|---|---|
| `gemma4` | `gemma4` | `Gemma4ForConditionalGeneration` | **has** ViT-style vision encoder + Conformer audio encoder + MoE |
| `gemma4_unified` | `gemma4_unified` | `Gemma4UnifiedForConditionalGeneration` | **encoder-free**, dense |

**The 12B-it checkpoint is `gemma4_unified`.** Do not benchmark/cite the `gemma4` module by
mistake — it is the encoder-*ful* sibling for other family members. "Encoder-free" is a
property of this checkpoint's architecture class + config, not of the Gemma 4 family as a whole.

## 1. Parameter budget (measured, meta-device build)

Total **11.96 B** params:
- `language_model` (decoder-only LLM): **11.91 B** — 99.56%
- `embed_vision` (entire image path): **49.9 M** — 0.42%
  - patch embedder (LN + Dense(6912→3840) + LN + pos-emb + LN): **35.2 M**
  - shared projection (`multimodal_embedder`: RMSNorm + Linear(3840→3840)): **14.7 M**
- `embed_audio` (entire audio path): **2.46 M** — 0.02% (just the shared-projection class, Linear(640→3840))
- `lm_head`: 1.01 B (tied embeddings; counted within/with the LM)

> **Reconciling Google's "~35M vision embedder" with our 49.9M.** Not a contradiction — a module
> boundary. Google counts only the **patch embedder** (the part that *replaces* the vision
> encoder) = **35.2M** (`35,176,704`). Our `embed_vision` module also contains the generic
> **projection into LM space** (`Gemma4UnifiedMultimodalEmbedder` = RMSNorm + Linear(3840→3840))
> = **14.7M**, which is the *same class* the audio path uses as its entire pipeline. So
> **49.9M = 35.2M (vision-specific) + 14.7M (shared projection)**, and audio = 2.46M is just that
> shared projection at 640→3840. These are the headline evidence that there is no encoder.

For the paper, phrase it as: *"the encoder-free vision adapter comprises a 35.2M-parameter patch
embedder plus a 14.7M shared projection into the LM space (49.9M total); audio reuses only the
2.46M projection stage."*

## 2. Vision path — `Gemma4UnifiedVisionEmbedder` (NO attention, NO transformer)

Raw pixels are patchified at `patch_size=16` then 3×3-merged (`pooling_kernel_size=3`) into
**`model_patch_size = 48`** px model-patches → each flattened to `48²·3 = 6912` values.

```
pixel patch (6912)
  → LayerNorm(6912)                      # patch_ln1
  → Linear(6912 → 3840)                  # patch_dense          [26.5M params]
  → LayerNorm(3840)                      # patch_ln2
  → + factorized 2D pos-emb              # pos_embedding (1120, 2, 3840)  [8.6M]
       pos = posemb[x, axis0] + posemb[y, axis1]
  → LayerNorm(3840)                      # pos_norm
  → RMSNorm(3840, no scale)              # multimodal_embedder.pre_projection_norm
  → Linear(3840 → 3840, no bias)         # multimodal_embedder.embedding_projection [14.7M]
  → soft tokens in LLM space (3840)
```
`num_soft_tokens = 280` per image (after pooling). Padding patches (pos_id = -1) are
stripped before being scattered into the sequence.

## 3. Audio path — just `Gemma4UnifiedMultimodalEmbedder` (one Linear)

Feature extractor reshapes raw 16 kHz waveform into frames of `audio_samples_per_token = 640`
samples = **40 ms each**. One frame = one soft token.

```
audio frame (640 raw samples)
  → RMSNorm(640, no scale)               # embedding_pre_projection_norm   [0 params]
  → Linear(640 → 3840, no bias)          # embedding_projection            [2.46M]
  → soft token in LLM space (3840)
```
No conv stack, no attention, no Conformer. This is the most minimal "encoder" possible.

## 4. How modalities enter the LLM

`Gemma4UnifiedModel.forward`:
1. Text → `embed_tokens` (scaled word embedding, ×√hidden).
2. Image/audio placeholder tokens (`image_token_id=258880`, `audio_token_id=258881`) are
   located by `get_placeholder_mask`.
3. Vision/audio soft tokens are computed and `masked_scatter`-ed into those positions.
4. The merged `(batch, seq, 3840)` stream goes through the **same** 48 decoder layers.
   → This is the "unified" claim: one transformer processes all modalities.

## 5. The decoder-only LLM (`Gemma4UnifiedTextModel`)

- hidden 3840, **48 layers**, intermediate 15360, vocab 262144, ctx 262144 (256K).
- **GQA**: 16 query heads / 8 KV heads.
- **Hybrid attention**: `layer_types` = 40 `sliding_attention` (window 1024) + 8 `full_attention`,
  pattern 5:1, **last layer forced to full_attention**.
- **p-RoPE**: full-attention layers use `rope_type=proportional`, `partial_rotary_factor=0.25`,
  θ=1e6; sliding layers use default RoPE θ=1e4.
- **QK-norm**: per-head RMSNorm on Q, K, and V (V-norm has no scale).
- **attention_k_eq_v = True**: on global layers K and V **share** the same projection
  (`v_proj is None`, `value_states = key_states`) with `global_head_dim=512`.
- **num_kv_shared_layers = 0**: cross-layer KV sharing is *supported by code* but OFF in this checkpoint.
- Decoder layer = Gemma-style 4-norm sandwich (input / post-attn / pre-ff / post-ff RMSNorm)
  + a per-layer `layer_scalar` buffer, residual around attn and MLP.
- `use_bidirectional_attention = "vision"`: image soft-token blocks attend bidirectionally
  (within a block), text stays causal.

## 6. Research angles this architecture opens (for CVPR framing)

- **Where does "vision understanding" live if there's no vision encoder?** Probe which of the
  48 layers first linearly-decode object/scene attributes from image soft tokens (linear probes
  per layer → the `hidden_states_per_layer` trace from script 02 is the substrate).
- **Soft-token geometry**: image (280) vs audio (per-40ms) vs text tokens occupy the same 3840
  space — compare their norm/anisotropy/CKA across depth.
- **Ablations**: the entire vision encoder is 3 LayerNorms + 2 Linears + a pos-emb table; trivially
  swappable. Great for "what is the minimal viable modality adapter?" studies.
- **p-RoPE + K=V global attention** is unusual; worth an attention-pattern analysis on long
  multimodal context.

## 7. Reproduce
```
.venv/bin/python scripts/01_architecture_report.py   # static, no weights needed
.venv/bin/python scripts/02_modality_trace.py         # per-layer / per-modality shapes (needs weights)
```
