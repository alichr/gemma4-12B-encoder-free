#!/usr/bin/env python
"""
04_representation_geometry.py — How do text / image / audio tokens live in the shared space?

Because Gemma 4 12B is encoder-free and "unified", text, image, and audio tokens all flow
through the SAME 48 decoder layers in the same 3840-dim space. This script characterizes the
representation geometry per modality, per layer:

  1. token L2-norm          (activation growth / outlier behaviour per modality)
  2. intra-modality anisotropy (mean pairwise cosine — representation spread/collapse)
  3. cross-modal alignment  (cosine between modality centroids — do modalities converge?)
  4. layer-vs-layer linear CKA of image tokens (representational evolution with depth)

Outputs four figures + JSON.
"""
import json
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import (load, build_inputs, hidden_states_and_masks, make_shape_image,
                     sine_audio, COLORS, SHAPES, FIG_DIR, DEVICE)

MODS = ["text", "image", "audio"]
COL = {"text": "#2a9d8f", "image": "#e76f51", "audio": "#264653"}


def linear_cka(X, Y):
    """Linear CKA between two (n, d) feature matrices with matched rows."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    hsic = np.linalg.norm(Y.T @ X, "fro") ** 2
    return hsic / (np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro") + 1e-12)


def mean_pairwise_cos(A, cap=256, seed=0):
    if len(A) > cap:
        A = A[np.random.default_rng(seed).choice(len(A), cap, replace=False)]
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    S = An @ An.T
    n = len(An)
    return (S.sum() - np.trace(S)) / (n * (n - 1) + 1e-8)


def main():
    model, processor = load()
    image_id, audio_id = model.config.image_token_id, model.config.audio_token_id
    n_layers = model.config.text_config.num_hidden_layers + 1

    # --- varied multimodal samples ---
    color_names = list(COLORS.keys())
    specs = [("circle", "red", 220), ("square", "blue", 330), ("triangle", "green", 440),
             ("circle", "yellow", 550), ("square", "magenta", 660), ("triangle", "cyan", 180)]

    # collectors[L][modality] -> list of (n_i, D) arrays
    collected = {L: {m: [] for m in MODS} for L in range(n_layers)}

    print("Collecting per-modality token representations...")
    for k, (shape, color, freq) in enumerate(specs):
        img = make_shape_image(shape, color, seed=1000 + k)
        aud = sine_audio(freq)
        inp = build_inputs(processor, image=img, audio=aud,
                           text=f"Describe the {shape} and the sound.")
        hs, masks = hidden_states_and_masks(model, inp, image_id, audio_id)
        for L, h in enumerate(hs):
            for m in MODS:
                v = h[0][masks[m][0]].float().cpu().numpy()
                if len(v):
                    collected[L][m].append(v)
        print(f"  sample {k+1}/{len(specs)}")

    # concat
    reps = {L: {m: (np.concatenate(collected[L][m], 0) if collected[L][m] else np.zeros((0, 1)))
                for m in MODS} for L in range(n_layers)}

    # --- metrics ---
    norms = {m: [float(np.linalg.norm(reps[L][m], axis=1).mean()) for L in range(n_layers)] for m in MODS}
    aniso = {m: [float(mean_pairwise_cos(reps[L][m])) for L in range(n_layers)] for m in MODS}
    centroids = {L: {m: reps[L][m].mean(0) for m in MODS} for L in range(n_layers)}

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    cross = {
        "image-text": [cos(centroids[L]["image"], centroids[L]["text"]) for L in range(n_layers)],
        "image-audio": [cos(centroids[L]["image"], centroids[L]["audio"]) for L in range(n_layers)],
        "audio-text": [cos(centroids[L]["audio"], centroids[L]["text"]) for L in range(n_layers)],
    }

    # --- CKA across layers for image tokens (matched rows) ---
    n_img = reps[0]["image"].shape[0]
    sub = np.random.default_rng(0).choice(n_img, min(600, n_img), replace=False)
    img_by_layer = [reps[L]["image"][sub] for L in range(n_layers)]
    cka = np.zeros((n_layers, n_layers), dtype=np.float32)
    for i in range(n_layers):
        for j in range(i, n_layers):
            c = linear_cka(img_by_layer[i], img_by_layer[j])
            cka[i, j] = cka[j, i] = c

    results = {"layers": list(range(n_layers)), "norms": norms, "anisotropy": aniso,
               "cross_modal_cos": cross, "cka_image": cka.tolist()}
    with open(os.path.join(FIG_DIR, "..", "representation_geometry.json"), "w") as f:
        json.dump(results, f, indent=2)

    x = list(range(n_layers))
    # Fig 1: norms
    fig, ax = plt.subplots(figsize=(9, 5))
    for m in MODS:
        ax.plot(x, norms[m], "-o", ms=3, lw=2, color=COL[m], label=m)
    ax.set_xlabel("layer"); ax.set_ylabel("mean token L2 norm")
    ax.set_title("Per-modality activation norm across depth"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_token_norm.png"), dpi=150); plt.close(fig)

    # Fig 2: anisotropy
    fig, ax = plt.subplots(figsize=(9, 5))
    for m in MODS:
        ax.plot(x, aniso[m], "-o", ms=3, lw=2, color=COL[m], label=m)
    ax.set_xlabel("layer"); ax.set_ylabel("mean pairwise cosine (intra-modality)")
    ax.set_title("Representation anisotropy per modality"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_anisotropy.png"), dpi=150); plt.close(fig)

    # Fig 3: cross-modal alignment
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, v in cross.items():
        ax.plot(x, v, "-o", ms=3, lw=2, label=k)
    ax.set_xlabel("layer"); ax.set_ylabel("cosine between modality centroids")
    ax.set_title("Cross-modal alignment across depth"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_cross_modal.png"), dpi=150); plt.close(fig)

    # Fig 4: CKA heatmap
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cka, origin="lower", cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("layer"); ax.set_ylabel("layer")
    ax.set_title("Linear CKA of image-token representations\n(layer i vs layer j)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="CKA")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_cka_image.png"), dpi=150); plt.close(fig)

    print("Wrote 4 figures + representation_geometry.json")
    print(f"GPU mem: {torch.cuda.max_memory_allocated(DEVICE)/1e9:.1f} GB")


if __name__ == "__main__":
    main()
