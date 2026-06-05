```
# Gemma 4 12B-it — architecture report

model_type        : gemma4_unified
architectures     : ['Gemma4UnifiedForConditionalGeneration']

## Total parameters: 11.96B (11,959,730,176)

## Parameter budget by component
  language_model (decoder-only LLM)         :    11.91B  (99.56% of total)
  embed_vision  (encoder-free image path)   :    49.92M  ( 0.42% of total)
  embed_audio   (encoder-free audio path)   :     2.46M  ( 0.02% of total)
  lm_head                                   :     1.01B  ( 8.42% of total)

## Vision embedder split (why 35M vs 49.9M)
  patch embedder (replaces vision encoder):    35.18M  <- Google's '~35M'
  shared projection into LLM space         :    14.75M  (same module class as audio)
  full embed_vision module                 :    49.92M

## Encoder-free verification
  vision_tower attribute present : False
  audio_tower  attribute present : False
  encoder-like submodule classes : NONE (confirmed encoder-free)
  vision_config.num_hidden_layers: ABSENT
  audio_config.num_hidden_layers : ABSENT

## Vision path (embed_vision) — module-by-module
  <root>                           Gemma4UnifiedVisionEmbedder {'pos_embedding': (1120, 2, 3840)}
  patch_ln1                        LayerNorm              {'weight': (6912,), 'bias': (6912,)}
  patch_dense                      Linear                 {'weight': (3840, 6912), 'bias': (3840,)}
  patch_ln2                        LayerNorm              {'weight': (3840,), 'bias': (3840,)}
  pos_norm                         LayerNorm              {'weight': (3840,), 'bias': (3840,)}
  multimodal_embedder.embedding_projection Linear                 {'weight': (3840, 3840)}
  -- all parameters --
     pos_embedding                            (1120, 2, 3840)
     patch_ln1.weight                         (6912,)
     patch_ln1.bias                           (6912,)
     patch_dense.weight                       (3840, 6912)
     patch_dense.bias                         (3840,)
     patch_ln2.weight                         (3840,)
     patch_ln2.bias                           (3840,)
     pos_norm.weight                          (3840,)
     pos_norm.bias                            (3840,)
     multimodal_embedder.embedding_projection.weight (3840, 3840)

## Audio path (embed_audio) — module-by-module
     embedding_projection.weight              (3840, 640)

## Text LLM (language_model)
  text.hidden_size                = 3840
  text.num_hidden_layers          = 48
  text.num_attention_heads        = 16
  text.num_key_value_heads        = 8
  text.head_dim                   = 256
  text.global_head_dim            = 512
  text.intermediate_size          = 15360
  text.vocab_size                 = 262144
  text.sliding_window             = 1024
  text.num_kv_shared_layers       = 0
  text.attention_k_eq_v           = True
  text.use_bidirectional_attention = vision
  text.max_position_embeddings    = 262144
  layer_types pattern        = ['sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention']
  layer_types counts         = {'sliding_attention': 40, 'full_attention': 8}
  rope_parameters            = {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"}, "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}}

## Encoder-free config knobs
  vision.patch_size             = 16
  vision.pooling_kernel_size    = 3
  vision.model_patch_size       = 48
  vision.mm_embed_dim           = 3840
  vision.mm_posemb_size         = 1120
  vision.num_soft_tokens        = 280
  vision.output_proj_dims       = 3840
  audio.audio_samples_per_token = 640
  audio.audio_embed_dim         = 640
  audio.hidden_size             = 640
  audio.output_proj_dims        = 640

(See notes/architecture_report.md for the saved copy.)
```
