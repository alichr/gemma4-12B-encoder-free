"""Shared helpers for Gemma 4 12B encoder-free analysis scripts."""
import os
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForMultimodalLM

HERE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(HERE, "..", "models", "gemma-4-12B-it")
FIG_DIR = os.path.join(HERE, "..", "outputs", "figures")
DEVICE = "cuda:0"

os.makedirs(FIG_DIR, exist_ok=True)

# Reproducible synthetic semantic dataset ------------------------------------
COLORS = {
    "red": (220, 30, 30), "green": (30, 180, 60), "blue": (40, 70, 220),
    "yellow": (230, 220, 40), "magenta": (210, 40, 200), "cyan": (40, 200, 210),
}
SHAPES = ["circle", "square", "triangle"]


def make_shape_image(shape: str, color: str, size: int = 224, seed: int = 0) -> Image.Image:
    """A shape of a given color on a neutral gray background, jittered for variety."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (size, size), (128, 128, 128))
    d = ImageDraw.Draw(img)
    m = size // 5
    jx, jy = int(rng.integers(-m // 2, m // 2)), int(rng.integers(-m // 2, m // 2))
    box = [m + jx, m + jy, size - m + jx, size - m + jy]
    c = COLORS[color]
    if shape == "circle":
        d.ellipse(box, fill=c)
    elif shape == "square":
        d.rectangle(box, fill=c)
    else:  # triangle
        d.polygon([(box[0], box[3]), (box[2], box[3]), ((box[0] + box[2]) // 2, box[1])], fill=c)
    return img


def sine_audio(freq: float, sr: int = 16000, dur: float = 1.0) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# Model ----------------------------------------------------------------------
def load(device: str = DEVICE):
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map=device
    ).eval()
    return model, processor


def build_inputs(processor, image=None, audio=None, text="Describe the input.", device=DEVICE):
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    if audio is not None:
        content.append({"type": "audio", "audio": audio})
    content.append({"type": "text", "text": text})
    msgs = [{"role": "user", "content": content}]
    return processor.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(device)


@torch.no_grad()
def hidden_states_and_masks(model, inputs, image_id, audio_id):
    """Return (hidden_states tuple, masks dict) for one batch.
    masks: per-modality boolean (B, S) over valid (attended) positions."""
    out = model(**inputs, output_hidden_states=True, use_cache=False)
    ids = inputs["input_ids"]
    attn = inputs.get("attention_mask", torch.ones_like(ids)).bool()
    img_m = (ids == image_id) & attn
    aud_m = (ids == audio_id) & attn
    txt_m = (~img_m) & (~aud_m) & attn
    return out.hidden_states, {"text": txt_m, "image": img_m, "audio": aud_m}
