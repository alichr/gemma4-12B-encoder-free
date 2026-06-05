#!/usr/bin/env python
"""
01_architecture_report.py — Static architecture report for Gemma 4 12B (encoder-free).

Runs WITHOUT downloaded weights: it builds the model on the `meta` device from the
config alone, so parameter counts and the module tree are exact while using ~0 RAM/VRAM.

Outputs:
  - prints a human-readable report
  - writes notes/architecture_report.md

Purpose for the research: empirically establish *what makes this model "encoder-free"*
by showing (a) there is no vision/audio transformer encoder module, and (b) the entire
non-text path is a handful of LayerNorm/Linear/RMSNorm layers.
"""
import json
import os
import sys
import torch
from transformers import AutoConfig, AutoModelForMultimodalLM

REPO = "google/gemma-4-12B-it"
LOCAL = os.path.join(os.path.dirname(__file__), "..", "models", "gemma-4-12B-it")
SRC = LOCAL if os.path.isdir(LOCAL) else REPO


def human(n: int) -> str:
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000
    return f"{n:.2f}T"


def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def main():
    config = AutoConfig.from_pretrained(SRC)
    print(f"# Gemma 4 12B-it — architecture report\n")
    print(f"model_type        : {config.model_type}")
    print(f"architectures     : {config.architectures}")

    tc, vc, ac = config.text_config, config.vision_config, config.audio_config

    # Build on meta device: constructs every nn.Module (so param shapes are real) with no allocation.
    with torch.device("meta"):
        model = AutoModelForMultimodalLM.from_config(config)

    total = count_params(model)
    print(f"\n## Total parameters: {human(total)} ({total:,})\n")

    # --- top-level component breakdown ---
    print("## Parameter budget by component")
    base = model.model if hasattr(model, "model") else model
    components = {
        "language_model (decoder-only LLM)": getattr(base, "language_model", None),
        "embed_vision  (encoder-free image path)": getattr(base, "embed_vision", None),
        "embed_audio   (encoder-free audio path)": getattr(base, "embed_audio", None),
        "lm_head": getattr(model, "lm_head", None),
    }
    for name, mod in components.items():
        if mod is None:
            print(f"  {name:<42}: ABSENT")
            continue
        n = count_params(mod)
        print(f"  {name:<42}: {human(n):>9}  ({100*n/total:5.2f}% of total)")

    # --- vision embedder split: encoder-replacement vs shared LLM projection ---
    # Reconciles Google's "~35M vision embedder" with the full embed_vision module.
    ev_split = getattr(base, "embed_vision", None)
    if ev_split is not None:
        patch = sum(p.numel() for nm, p in ev_split.named_parameters()
                    if not nm.startswith("multimodal_embedder"))
        proj = sum(p.numel() for nm, p in ev_split.named_parameters()
                   if nm.startswith("multimodal_embedder"))
        print("\n## Vision embedder split (why 35M vs 49.9M)")
        print(f"  patch embedder (replaces vision encoder): {human(patch):>9}  <- Google's '~35M'")
        print(f"  shared projection into LLM space         : {human(proj):>9}  (same module class as audio)")
        print(f"  full embed_vision module                 : {human(patch+proj):>9}")

    # --- the encoder-free proof ---
    print("\n## Encoder-free verification")
    mod_names = [type(m).__name__ for m in model.modules()]
    encoder_like = sorted({n for n in mod_names if any(
        k in n for k in ["Encoder", "VisionAttention", "AudioAttention", "Conformer", "SubSample", "EncoderLayer"]
    )})
    print(f"  vision_tower attribute present : {hasattr(base, 'vision_tower')}")
    print(f"  audio_tower  attribute present : {hasattr(base, 'audio_tower')}")
    print(f"  encoder-like submodule classes : {encoder_like if encoder_like else 'NONE (confirmed encoder-free)'}")
    print(f"  vision_config.num_hidden_layers: {getattr(vc, 'num_hidden_layers', 'ABSENT')}")
    print(f"  audio_config.num_hidden_layers : {getattr(ac, 'num_hidden_layers', 'ABSENT')}")

    # --- vision path submodules ---
    print("\n## Vision path (embed_vision) — module-by-module")
    ev = getattr(base, "embed_vision", None)
    if ev is not None:
        for n, m in ev.named_modules():
            if list(m.parameters(recurse=False)) or isinstance(m, torch.nn.Parameter):
                shapes = {pn: tuple(p.shape) for pn, p in m.named_parameters(recurse=False)}
                if shapes:
                    print(f"  {n or '<root>':<32} {type(m).__name__:<22} {shapes}")
        # named parameters (catches the bare nn.Parameter pos_embedding)
        print("  -- all parameters --")
        for pn, p in ev.named_parameters():
            print(f"     {pn:<40} {tuple(p.shape)}")

    # --- audio path submodules ---
    print("\n## Audio path (embed_audio) — module-by-module")
    ea = getattr(base, "embed_audio", None)
    if ea is not None:
        for pn, p in ea.named_parameters():
            print(f"     {pn:<40} {tuple(p.shape)}")

    # --- text/LLM structure ---
    print("\n## Text LLM (language_model)")
    for k in ["hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
              "head_dim", "global_head_dim", "intermediate_size", "vocab_size", "sliding_window",
              "num_kv_shared_layers", "attention_k_eq_v", "use_bidirectional_attention",
              "max_position_embeddings"]:
        if hasattr(tc, k):
            print(f"  text.{k:<26} = {getattr(tc, k)}")
    lt = getattr(tc, "layer_types", None)
    if lt:
        from collections import Counter
        print(f"  layer_types pattern        = {lt}")
        print(f"  layer_types counts         = {dict(Counter(lt))}")
    rp = getattr(tc, "rope_parameters", None)
    if rp:
        print(f"  rope_parameters            = {json.dumps(rp)}")

    # --- vision/audio config knobs that define the encoder-free embedding ---
    print("\n## Encoder-free config knobs")
    for k in ["patch_size", "pooling_kernel_size", "model_patch_size", "mm_embed_dim",
              "mm_posemb_size", "num_soft_tokens", "output_proj_dims"]:
        if hasattr(vc, k):
            print(f"  vision.{k:<22} = {getattr(vc, k)}")
    for k in ["audio_samples_per_token", "audio_embed_dim", "hidden_size", "output_proj_dims"]:
        if hasattr(ac, k):
            print(f"  audio.{k:<23} = {getattr(ac, k)}")

    print("\n(See notes/architecture_report.md for the saved copy.)")


if __name__ == "__main__":
    # tee stdout to the markdown file
    out_path = os.path.join(os.path.dirname(__file__), "..", "notes", "architecture_report.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, d):
            for s in self.streams:
                s.write(d)
        def flush(self):
            for s in self.streams:
                s.flush()

    with open(out_path, "w") as f:
        f.write("```\n")
        sys.stdout = Tee(sys.__stdout__, f)
        main()
        f.write("```\n")
    sys.stdout = sys.__stdout__
    print(f"\nWrote {out_path}")
