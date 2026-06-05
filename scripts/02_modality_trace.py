#!/usr/bin/env python
"""
02_modality_trace.py — Per-modality, per-layer activation tracer for Gemma 4 12B.

Loads the real weights on one GPU (bf16), feeds a single prompt containing TEXT + IMAGE
+ AUDIO, and uses forward hooks to record the shape/dtype of every module's output.

It answers the question you asked directly:
  - what dimension does each modality have at the input,
  - how each modality is projected (encoder-free) into the LLM's 3840-dim space,
  - what the hidden state looks like at every one of the 48 decoder layers,
  - which sequence positions are text vs image vs audio soft tokens.

Outputs:
  - prints readable traces
  - writes outputs/modality_trace.json   (machine-readable, for plotting in the paper)
"""
import json
import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM

HERE = os.path.dirname(__file__)
LOCAL = os.path.join(HERE, "..", "models", "gemma-4-12B-it")
OUT = os.path.join(HERE, "..", "outputs", "modality_trace.json")
DEVICE = "cuda:0"


def shp(x):
    if torch.is_tensor(x):
        return {"shape": tuple(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, (list, tuple)):
        return [shp(v) for v in x]
    return str(type(x).__name__)


def main():
    record = {}

    print("Loading processor + model (bf16, single GPU)...")
    processor = AutoProcessor.from_pretrained(LOCAL)
    model = AutoModelForMultimodalLM.from_pretrained(
        LOCAL, dtype=torch.bfloat16, device_map=DEVICE
    ).eval()

    cfg = model.config
    image_token_id = cfg.image_token_id
    audio_token_id = cfg.audio_token_id

    # --- synthetic, reproducible inputs (no downloads) ---
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
    # 1.0 s of 16 kHz audio = 16000 samples -> 25 frames of 640 -> 25 audio soft tokens
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "audio", "audio": audio},
            {"type": "text", "text": "Describe what you see and hear."},
        ],
    }]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(DEVICE)

    print("\n=== PROCESSOR OUTPUT (per-modality input tensors) ===")
    record["inputs"] = {}
    for k, v in inputs.items():
        record["inputs"][k] = shp(v)
        print(f"  {k:<22} {shp(v)}")

    ids = inputs["input_ids"][0]
    n_img = int((ids == image_token_id).sum())
    n_aud = int((ids == audio_token_id).sum())
    n_txt = int(ids.numel() - n_img - n_aud)
    record["token_type_counts"] = {"total": int(ids.numel()), "text": n_txt,
                                   "image_soft_tokens": n_img, "audio_soft_tokens": n_aud}
    print(f"\n  sequence length          : {ids.numel()}")
    print(f"  text (hard) tokens       : {n_txt}")
    print(f"  image soft tokens        : {n_img}  (id={image_token_id})")
    print(f"  audio soft tokens        : {n_aud}  (id={audio_token_id})")

    # --- register hooks on every named module ---
    captured = {}
    handles = []

    def make_hook(name):
        def hook(mod, args, out):
            captured[name] = {"in": [shp(a) for a in args], "out": shp(out)}
        return hook

    for name, mod in model.named_modules():
        if name:  # skip root
            handles.append(mod.register_forward_hook(make_hook(name)))

    print("\nRunning a single multimodal forward (prefill, output_hidden_states=True)...")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)

    for h in handles:
        h.remove()

    # --- vision (encoder-free) path trace ---
    print("\n=== VISION PATH (embed_vision) — encoder-free pipeline ===")
    vis_order = ["model.embed_vision.patch_ln1", "model.embed_vision.patch_dense",
                 "model.embed_vision.patch_ln2", "model.embed_vision.pos_norm",
                 "model.embed_vision.multimodal_embedder.embedding_pre_projection_norm",
                 "model.embed_vision.multimodal_embedder.embedding_projection",
                 "model.embed_vision"]
    record["vision_path"] = {}
    for n in vis_order:
        if n in captured:
            record["vision_path"][n] = captured[n]
            print(f"  {n.replace('model.embed_vision','EV'):<48} out={captured[n]['out']}")

    # --- audio (encoder-free) path trace ---
    print("\n=== AUDIO PATH (embed_audio) — encoder-free pipeline ===")
    record["audio_path"] = {}
    for n in sorted(k for k in captured if k.startswith("model.embed_audio")):
        record["audio_path"][n] = captured[n]
        print(f"  {n.replace('model.embed_audio','EA'):<48} out={captured[n]['out']}")

    # --- per decoder layer hidden states ---
    print("\n=== PER-LAYER HIDDEN STATES (unified text+image+audio stream) ===")
    hs = out.hidden_states  # tuple: embeddings + each layer
    record["hidden_states_per_layer"] = []
    lt = getattr(cfg.text_config, "layer_types", [None] * (len(hs) - 1))
    for i, h in enumerate(hs):
        tag = "embeddings" if i == 0 else f"layer {i-1:>2} [{lt[i-1]}]"
        stats = {"index": i, "tag": tag, "shape": tuple(h.shape),
                 "mean": float(h.float().mean()), "std": float(h.float().std()),
                 "absmax": float(h.float().abs().max())}
        record["hidden_states_per_layer"].append(stats)
        if i == 0 or i % 6 == 0 or i == len(hs) - 1:
            print(f"  {tag:<26} shape={tuple(h.shape)}  mean={stats['mean']:+.4f} "
                  f"std={stats['std']:.4f} absmax={stats['absmax']:.1f}")

    print(f"\n  (full per-layer stats for all {len(hs)} entries saved to JSON)")
    print(f"  final logits shape       : {tuple(out.logits.shape)}")

    record["logits_shape"] = tuple(out.logits.shape)
    with open(OUT, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nWrote {OUT}")
    print(f"GPU mem allocated: {torch.cuda.max_memory_allocated(DEVICE)/1e9:.1f} GB")


if __name__ == "__main__":
    main()
