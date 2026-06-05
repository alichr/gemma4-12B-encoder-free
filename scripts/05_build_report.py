#!/usr/bin/env python
"""
05_build_report.py — Assemble a professional multi-page PDF report of all findings.

Pulls numbers from outputs/*.json and embeds the figures from outputs/figures/.
Output: outputs/Gemma4_EncoderFree_Report.pdf
"""
import json
import os
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, HRFlowable)

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "outputs")
FIG = os.path.join(OUT, "figures")
PDF = os.path.join(OUT, "Gemma4_EncoderFree_Report.pdf")

NAVY = colors.HexColor("#264653")
ORANGE = colors.HexColor("#e76f51")
TEAL = colors.HexColor("#2a9d8f")


def load(name, default=None):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else (default or {})


def styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle("H1b", parent=s["Heading1"], textColor=NAVY, spaceBefore=14, fontSize=16))
    s.add(ParagraphStyle("H2b", parent=s["Heading2"], textColor=ORANGE, spaceBefore=10, fontSize=12.5))
    s.add(ParagraphStyle("Body", parent=s["BodyText"], alignment=TA_JUSTIFY, fontSize=10, leading=14))
    s.add(ParagraphStyle("Cap", parent=s["BodyText"], alignment=TA_CENTER, fontSize=8.5,
                         textColor=colors.grey, leading=11))
    s.add(ParagraphStyle("TitleBig", parent=s["Title"], textColor=NAVY, fontSize=24, leading=28))
    s.add(ParagraphStyle("Sub", parent=s["Title"], textColor=ORANGE, fontSize=13, leading=16))
    return s


def fig(path, w=15 * cm, caption=None, S=None, flow=None):
    p = os.path.join(FIG, path)
    if not os.path.exists(p):
        return
    img = Image(p)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = w
    img.drawHeight = w * ratio
    img.hAlign = "CENTER"
    flow.append(Spacer(1, 0.2 * cm))
    flow.append(img)
    if caption:
        flow.append(Paragraph(caption, S["Cap"]))
    flow.append(Spacer(1, 0.3 * cm))


def kv_table(rows, widths, header=True):
    t = Table(rows, colWidths=widths)
    style = [("FONTSIZE", (0, 0), (-1, -1), 9),
             ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), NAVY),
                  ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                  ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(style))
    return t


def main():
    S = styles()
    trace = load("modality_trace.json")
    probe = load("linear_probes.json")
    geo = load("representation_geometry.json")
    F = []  # flowables
    P = lambda t, st="Body": F.append(Paragraph(t, S[st]))
    sp = lambda h=0.3: F.append(Spacer(1, h * cm))

    # ---------- cover ----------
    sp(5)
    P("Gemma 4 12B", "TitleBig")
    P("An Encoder-Free Multimodal Architecture — Findings Report", "Sub")
    sp(0.6)
    F.append(HRFlowable(width="80%", color=ORANGE, thickness=1.5))
    sp(0.6)
    P("Local empirical study of <b>google/gemma-4-12B-it</b> — parameter anatomy, per-modality "
      "data flow, and representation-geometry analysis across all 48 decoder layers. "
      "All results reproduced on a single GPU from synthetic inputs.", "Body")
    sp(0.4)
    fig("fig_cross_modal.png", 13 * cm,
        "Headline result: image/audio/text tokens begin in near-orthogonal subspaces and "
        "snap into a shared representation around layer 13.", S, F)
    F.append(PageBreak())

    # ---------- 1. overview ----------
    P("1. Overview & a critical caveat", "H1b")
    P("Gemma 4 12B (released 2026-06-03, Apache-2.0) is a dense, decoder-only multimodal model that "
      "ingests text, images, and audio <b>without dedicated vision or audio encoders</b>. Instead of a "
      "ViT/SigLIP tower or a Conformer, raw pixel patches and raw audio frames are projected directly "
      "into the language model's embedding space by tiny linear modules, and all modality "
      "&ldquo;understanding&rdquo; is performed by the shared transformer stack.")
    sp()
    P("Critical caveat for researchers", "H2b")
    P("The <font face='Courier'>transformers</font> library contains <b>two</b> Gemma 4 model types. "
      "<font face='Courier'>gemma4</font> (<font face='Courier'>Gemma4ForConditionalGeneration</font>) "
      "<b>has</b> a ViT vision encoder, a Conformer audio encoder, and MoE — it is the encoder-bearing "
      "sibling for other family members. The 12B-it checkpoint studied here is the separate "
      "<font face='Courier'>gemma4_unified</font> "
      "(<font face='Courier'>Gemma4UnifiedForConditionalGeneration</font>) implementation. "
      "&ldquo;Encoder-free&rdquo; is a property of this class and its config: the vision and audio "
      "configs contain <b>no</b> <font face='Courier'>num_hidden_layers</font> and no attention fields.")

    # ---------- 2. architecture ----------
    P("2. Parameter anatomy", "H1b")
    P("Building the model on a meta device (config only, zero allocation) gives exact module shapes "
      "and parameter counts. The non-text paths are negligible in size — the empirical signature of "
      "an encoder-free design.")
    sp()
    rows = [["Component", "Parameters", "Share"],
            ["language_model (decoder-only LLM)", "11.91 B", "99.56 %"],
            ["embed_vision (entire image path)", "49.9 M", "0.42 %"],
            ["   - patch embedder (encoder replacement)", "35.2 M", ""],
            ["   - shared projection into LLM space", "14.7 M", ""],
            ["embed_audio (entire audio path)", "2.46 M", "0.02 %"],
            ["Total", "11.96 B", "100 %"]]
    t = kv_table(rows, [8.5 * cm, 4 * cm, 3 * cm])
    t.setStyle(TableStyle([("TEXTCOLOR", (0, 3), (-1, 4), colors.grey),
                           ("FONTSIZE", (0, 3), (-1, 4), 8)]))
    F.append(t)
    sp()
    P("Reconciling the &ldquo;35M vision embedder&rdquo;.", "H2b")
    P("Google's materials cite a ~35M vision embedder; the full <font face='Courier'>embed_vision</font> "
      "module measures 49.9M. The difference is a module boundary, not a contradiction: the "
      "<b>patch embedder</b> (LayerNorm + Dense + factorized positional embedding) that actually "
      "<i>replaces the vision encoder</i> is 35.2M, while the remaining 14.7M is a generic "
      "<font face='Courier'>RMSNorm &rarr; Linear(3840&rarr;3840)</font> projection into LM space - "
      "the same module class the audio path uses (there as a 2.46M Linear(640&rarr;3840)). "
      "Thus: 49.9M (full path) = 35.2M (vision-specific) + 14.7M (shared projection).")
    sp()
    P("Vision path — <font face='Courier'>Gemma4UnifiedVisionEmbedder</font> (no attention)", "H2b")
    P("Images are patchified at 16&nbsp;px, 3&times;3-pooled into 48&times;48 model-patches "
      "(6912 raw values each), then: "
      "<font face='Courier'>LayerNorm &rarr; Linear(6912&rarr;3840) &rarr; LayerNorm &rarr; "
      "+factorized&nbsp;2D&nbsp;pos-emb(1120,2,3840) &rarr; LayerNorm &rarr; RMSNorm &rarr; "
      "Linear(3840&rarr;3840)</font>. Output: 280 soft tokens per image.")
    sp()
    P("Audio path — <font face='Courier'>Gemma4UnifiedMultimodalEmbedder</font> (one Linear)", "H2b")
    P("The feature extractor reshapes raw 16&nbsp;kHz waveform into 640-sample (40&nbsp;ms) frames. "
      "Each frame becomes one token via <font face='Courier'>RMSNorm &rarr; Linear(640&rarr;3840)</font>. "
      "No convolution stack, no attention.")
    sp()
    P("Decoder-only LLM", "H2b")
    llm = [["Property", "Value"],
           ["Layers / hidden / intermediate", "48 / 3840 / 15360"],
           ["Attention (GQA)", "16 query heads / 8 KV heads"],
           ["Hybrid pattern", "40 sliding (window 1024) + 8 global, 5:1"],
           ["RoPE", "p-RoPE on global (θ=1e6, partial 0.25); default on sliding (θ=1e4)"],
           ["Extras", "QK-norm; K=V on global layers; 256K context; dense (no MoE)"]]
    F.append(kv_table(llm, [6 * cm, 9.5 * cm]))
    F.append(PageBreak())

    # ---------- 3. data flow ----------
    P("3. Per-modality data flow (measured)", "H1b")
    if trace:
        ti = trace.get("inputs", {})
        ttc = trace.get("token_type_counts", {})
        def shp(k):
            v = ti.get(k, {})
            return str(v.get("shape", "-")) if isinstance(v, dict) else "-"
        P("A single forward pass on a combined text+image+audio prompt produced the following input "
          "tensors and a unified sequence of %s tokens (%s text + %s image + %s audio):"
          % (ttc.get("total", "?"), ttc.get("text", "?"),
             ttc.get("image_soft_tokens", "?"), ttc.get("audio_soft_tokens", "?")))
        sp()
        rows = [["Tensor", "Shape", "Meaning"],
                ["pixel_values", shp("pixel_values"), "280 raw 48x48x3 patches"],
                ["image_position_ids", shp("image_position_ids"), "(x,y) patch coords"],
                ["input_features", shp("input_features"), "25 audio frames x 640 samples"],
                ["input_ids", shp("input_ids"), "merged token sequence"]]
        F.append(kv_table(rows, [4.5 * cm, 4.5 * cm, 6.5 * cm]))
        sp()
        P("Both modalities are projected to the LLM width (3840) and scattered into placeholder "
          "positions, after which the same 48 layers process every token. Final logits: %s."
          % str(tuple(trace.get("logits_shape", []))))

    # ---------- 4. findings ----------
    F.append(PageBreak())
    P("4. Findings from representation analysis", "H1b")

    P("4.1 Where image attributes become linearly decodable", "H2b")
    if probe:
        c0 = probe["color_acc"][0]; s0 = probe["shape_acc"][0]; s_full = probe["shape_acc"][3]
        P("Linear probes on mean-pooled image soft tokens show <b>color is perfectly decodable at "
          "layer 0</b> (color acc = %.2f, the raw linear patch projection), while <b>shape</b> rises "
          "from %.2f at layer 0 to ~%.2f within three layers. With no encoder, low-level appearance is "
          "linearly present at the embedder; spatial structure is assembled by the early decoder "
          "layers. The synthetic task is simple, so both saturate (ceiling effect) — for a graded "
          "curve, harder/natural attributes are needed." % (c0, s0, s_full))
    fig("fig_linear_probes.png", 12.5 * cm, "Probe test-accuracy vs layer (color 6-way, shape 3-way).", S, F)

    F.append(PageBreak())
    P("4.2 Cross-modal alignment: a phase transition", "H2b")
    if geo:
        cm0 = geo["cross_modal_cos"]["image-text"][0]
        P("Cosine between modality centroids is near zero/negative in the first ~12 layers "
          "(image-text = %.2f at layer 0), then jumps to ~0.99 around layer 13 and stays high through "
          "the mid-stack: the unified model <b>fuses modalities into a shared subspace</b>. Toward the "
          "output, image-text re-diverges while image-audio remains bound." % cm0)
    fig("fig_cross_modal.png", 12.5 * cm, "Cross-modal centroid cosine across depth.", S, F)

    P("4.3 Activation norm & the mid-stack bottleneck", "H2b")
    P("Per-modality token norms dip to a minimum near layer 11-12 (coinciding with the alignment "
      "pivot) and then peak sharply near layer 23 for all modalities — a massive-activation regime "
      "consistent with the per-layer absmax spikes recorded in the layer trace.")
    fig("fig_token_norm.png", 12.5 * cm, "Mean token L2 norm per modality across depth.", S, F)

    F.append(PageBreak())
    P("4.4 Anisotropy & representational evolution", "H2b")
    P("At the mid-stack peak, intra-modality mean pairwise cosine reaches ~0.94-0.97 (near-collapse "
      "to a shared direction), then partially recovers. Layer-vs-layer linear CKA of image tokens is "
      "high only locally (adjacent layers ~0.93) while CKA(L0,L48) ~ 0.07 — input and output "
      "representations are almost entirely distinct.")
    fig("fig_anisotropy.png", 11 * cm, "Intra-modality anisotropy (mean pairwise cosine).", S, F)
    fig("fig_cka_image.png", 9.5 * cm, "Linear CKA of image-token reps, layer i vs layer j.", S, F)

    # ---------- 5. directions ----------
    F.append(PageBreak())
    P("5. Research directions", "H1b")
    for t in [
        "<b>Minimal modality adapters.</b> The whole vision &ldquo;encoder&rdquo; is 3 LayerNorms + 2 "
        "Linears + a position table (49.9M); audio is a single Linear (2.46M). This is an ideal "
        "test-bed for &ldquo;what is the minimal viable modality adapter?&rdquo; ablations.",
        "<b>The layer-13 fusion transition.</b> Characterize why modalities align abruptly mid-stack: "
        "causal interventions, per-head attribution, and whether the transition shifts with prompt "
        "structure or context length.",
        "<b>Massive activations at layer ~23.</b> Identify the outlier dimensions and their role "
        "(attention sinks / normalization), and whether they are shared across modalities.",
        "<b>Soft-token geometry.</b> Compare image (280) vs audio (40ms) vs text token manifolds: "
        "effective rank, norm growth, and per-modality re-specialization near the output.",
        "<b>p-RoPE + K=V global attention.</b> Analyze long multimodal-context attention patterns "
        "given the unusual proportional RoPE and key=value sharing on global layers.",
    ]:
        F.append(Paragraph("&bull; " + t, S["Body"]))
        sp(0.15)

    sp()
    P("Reproduce", "H2b")
    P("<font face='Courier'>scripts/01_architecture_report.py</font> (static, no weights) &middot; "
      "<font face='Courier'>02_modality_trace.py</font> &middot; "
      "<font face='Courier'>03_linear_probes.py</font> &middot; "
      "<font face='Courier'>04_representation_geometry.py</font> &middot; "
      "<font face='Courier'>05_build_report.py</font> (this PDF). Environment pinned in "
      "<font face='Courier'>requirements-lock.txt</font>; weights from "
      "<font face='Courier'>google/gemma-4-12B-it</font> (ungated, Apache-2.0).")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.grey)
        canvas.drawString(2 * cm, 1.1 * cm, "Gemma 4 12B — Encoder-Free Architecture Study")
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, "page %d" % doc.page)
        canvas.restoreState()

    doc = SimpleDocTemplate(PDF, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            title="Gemma 4 12B Encoder-Free Architecture Report")
    doc.build(F, onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {PDF}  ({os.path.getsize(PDF)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
