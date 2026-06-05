#!/usr/bin/env python
"""
03_linear_probes.py — Where do image semantics emerge in an encoder-free model?

The vision path has NO encoder, so any "understanding" of an image must be built by the
shared LLM layers. This script localizes *where*: it trains a linear probe on the
mean-pooled image soft-token representation at every layer to predict two attributes of a
controlled synthetic dataset:
  - COLOR  (6 classes) — a low-level attribute
  - SHAPE  (3 classes) — a mid-level/semantic attribute

Probe test-accuracy vs layer tells you at which depth each attribute becomes linearly
decodable. Outputs a figure + JSON.
"""
import json
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from _common import load, build_inputs, hidden_states_and_masks, make_shape_image, COLORS, SHAPES, FIG_DIR, DEVICE

N_PER_COMBO = 12          # images per (shape,color) combo  -> 6*3*12 = 216 images
BATCH = 12
PROMPT = "What is in this image?"


def cat_inputs(input_list):
    keys = input_list[0].keys()
    return {k: torch.cat([d[k] for d in input_list], dim=0) for k in keys}


def main():
    model, processor = load()
    image_id, audio_id = model.config.image_token_id, model.config.audio_token_id

    # --- build controlled dataset ---
    samples = []  # (image, color_idx, shape_idx)
    color_names, shape_names = list(COLORS.keys()), SHAPES
    seed = 0
    for ci, color in enumerate(color_names):
        for si, shape in enumerate(shape_names):
            for _ in range(N_PER_COMBO):
                samples.append((make_shape_image(shape, color, seed=seed), ci, si))
                seed += 1

    n_layers = model.config.text_config.num_hidden_layers + 1  # + embeddings
    feats = np.zeros((len(samples), n_layers, model.config.text_config.hidden_size), dtype=np.float32)
    y_color = np.array([s[1] for s in samples])
    y_shape = np.array([s[2] for s in samples])

    print(f"Extracting per-layer image-token features for {len(samples)} images...")
    idx = 0
    for b in range(0, len(samples), BATCH):
        batch = samples[b:b + BATCH]
        inp = cat_inputs([build_inputs(processor, image=im, text=PROMPT) for im, _, _ in batch])
        hs, masks = hidden_states_and_masks(model, inp, image_id, audio_id)
        img_mask = masks["image"]  # (B,S)
        for L, h in enumerate(hs):  # h: (B,S,D)
            for j in range(len(batch)):
                m = img_mask[j]
                feats[idx + j, L] = h[j][m].float().mean(0).cpu().numpy()
        idx += len(batch)
        print(f"  {idx}/{len(samples)}")

    # --- probe each layer ---
    results = {"layers": list(range(n_layers)), "color_acc": [], "shape_acc": [],
               "color_chance": 1 / len(color_names), "shape_chance": 1 / len(shape_names)}

    def probe(X, y):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Xtr), ytr)
        return clf.score(sc.transform(Xte), yte)

    print("Training probes per layer...")
    for L in range(n_layers):
        ca = probe(feats[:, L], y_color)
        sa = probe(feats[:, L], y_shape)
        results["color_acc"].append(ca)
        results["shape_acc"].append(sa)
        if L % 6 == 0 or L == n_layers - 1:
            print(f"  layer {L:>2}: color={ca:.3f}  shape={sa:.3f}")

    with open(os.path.join(FIG_DIR, "..", "linear_probes.json"), "w") as f:
        json.dump(results, f, indent=2)

    # --- figure ---
    x = results["layers"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, results["color_acc"], "-o", ms=3, lw=2, color="#d1495b", label="color (6-way)")
    ax.plot(x, results["shape_acc"], "-o", ms=3, lw=2, color="#30638e", label="shape (3-way)")
    ax.axhline(results["color_chance"], ls="--", c="#d1495b", alpha=.5, lw=1, label="color chance")
    ax.axhline(results["shape_chance"], ls="--", c="#30638e", alpha=.5, lw=1, label="shape chance")
    ax.set_xlabel("layer (0 = patch embeddings, 1..48 = decoder layers)")
    ax.set_ylabel("linear-probe test accuracy")
    ax.set_title("Where image attributes become linearly decodable\n(encoder-free: all semantics built inside the LLM)")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_linear_probes.png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")
    print(f"GPU mem: {torch.cuda.max_memory_allocated(DEVICE)/1e9:.1f} GB")


if __name__ == "__main__":
    main()
