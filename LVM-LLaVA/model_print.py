LlavaLlamaForCausalLM(
  (model): LlavaLlamaModel(
    (embed_tokens): Embedding(32000, 4096, padding_idx=0)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaFlashAttention2(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)





CLIPVisionTower(
  (vision_tower): CLIPVisionModel(
    (vision_model): CLIPVisionTransformer(
      (embeddings): CLIPVisionEmbeddings(
        (patch_embedding): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14), bias=False)
        (position_embedding): Embedding(577, 1024)
      )
      (pre_layrnorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      (encoder): CLIPEncoder(
        (layers): ModuleList(
          (0-23): 24 x CLIPEncoderLayer(
            (self_attn): CLIPAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): QuickGELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
      (post_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    )
  )
)


LlavaLlamaForCausalLM(
  (model): LlavaLlamaModel(
    (embed_tokens): Embedding(32000, 4096, padding_idx=0)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaFlashAttention2(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
    (vision_tower): CLIPVisionTower(
      (vision_tower): CLIPVisionModel(
        (vision_model): CLIPVisionTransformer(
          (embeddings): CLIPVisionEmbeddings(
            (patch_embedding): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14), bias=False)
            (position_embedding): Embedding(577, 1024)
          )
          (pre_layrnorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (encoder): CLIPEncoder(
            (layers): ModuleList(
              (0-23): 24 x CLIPEncoderLayer(
                (self_attn): CLIPAttention(
                  (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
                  (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
                  (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
                  (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
                )
                (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (mlp): CLIPMLP(
                  (activation_fn): QuickGELUActivation()
                  (fc1): Linear(in_features=1024, out_features=4096, bias=True)
                  (fc2): Linear(in_features=4096, out_features=1024, bias=True)
                )
                (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              )
            )
          )
          (post_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    (mm_projector): Sequential(
      (0): Linear(in_features=1024, out_features=4096, bias=True)
      (1): GELU(approximate='none')
      (2): Linear(in_features=4096, out_features=4096, bias=True)
    )
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)




AIM(
  (preprocessor): ViTPreprocessor(
    (patchifier): PatchEmbed(
      (proj): Conv2d(3, 2048, kernel_size=(14, 14), stride=(14, 14))
      (norm): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
    )
    (pos_embed): SinCosPosEmbed()
  )
  (trunk): Transformer(
    (blocks): Sequential(
      (0): Block(
        (attn): Attention(
          (qkv): Linear(in_features=2048, out_features=6144, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=2048, out_features=2048, bias=False)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (norm_1): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
        (mlp): MLP(
          (fc1): Linear(in_features=2048, out_features=8192, bias=False)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=8192, out_features=2048, bias=False)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (norm_2): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
      )
      (1): Block(
        (attn): Attention(
          (qkv): Linear(in_features=2048, out_features=6144, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=2048, out_features=2048, bias=False)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (norm_1): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
        (mlp): MLP(
          (fc1): Linear(in_features=2048, out_features=8192, bias=False)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=8192, out_features=2048, bias=False)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (norm_2): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
      )
      (2): Block(
        (attn): Attention(
          (qkv): Linear(in_features=2048, out_features=6144, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=2048, out_features=2048, bias=False)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (norm_1): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
        (mlp): MLP(
          (fc1): Linear(in_features=2048, out_features=8192, bias=False)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=8192, out_features=2048, bias=False)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (norm_2): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
      )
      (3): Block(
        (attn): Attention(
          (qkv): Linear(in_features=2048, out_features=6144, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=2048, out_features=2048, bias=False)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (norm_1): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
        (mlp): MLP(
          (fc1): Linear(in_features=2048, out_features=8192, bias=False)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=8192, out_features=2048, bias=False)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (norm_2): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
      )
      (4): Block(
        (attn): Attention(
          (qkv): Linear(in_features=2048, out_features=6144, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=2048, out_features=2048, bias=False)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (norm_1): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
        (mlp): MLP(
          (fc1): Linear(in_features=2048, out_features=8192, bias=False)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=8192, out_features=2048, bias=False)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (norm_2): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
      )
      (5): Block(
        (attn): Attention(
          (qkv): Linear(in_features=2048, out_features=6144, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=2048, out_features=2048, bias=False)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (norm_1): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
        (mlp): MLP(
          (fc1): Linear(in_features=2048, out_features=8192, bias=False)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=8192, out_features=2048, bias=False)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (norm_2): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
      )
      (6): Block(
        (attn): Attention(
          (qkv): Linear(in_features=2048, out_features=6144, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=2048, out_features=2048, bias=False)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (norm_1): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
        (mlp): MLP(
          (fc1): Linear(in_features=2048, out_features=8192, bias=False)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=8192, out_features=2048, bias=False)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (norm_2): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
      )
      (7): Block(
        (attn): Attention(
          (qkv): Linear(in_features=2048, out_features=6144, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=2048, out_features=2048, bias=False)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (norm_1): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
        (mlp): MLP(
          (fc1): Linear(in_features=2048, out_features=8192, bias=False)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=8192, out_features=2048, bias=False)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (norm_2): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
      )
      (8): Block(
        (attn): Attention(
          (qkv): Linear(in_features=2048, out_features=6144, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=2048, out_features=2048, bias=False)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (norm_1): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
        (mlp): MLP(
          (fc1): Linear(in_features=2048, out_features=8192, bias=False)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=8192, out_features=2048, bias=False)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (norm_2): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
      )
      (9): Block(
        (attn): Attention(
          (qkv): Linear(in_features=2048, out_features=6144, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=2048, out_features=2048, bias=False)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (norm_1): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
        (mlp): MLP(
          (fc1): Linear(in_features=2048, out_features=8192, bias=False)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=8192, out_features=2048, bias=False)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (norm_2): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
      )
      (10): Block(
        (attn): Attention(
          (qkv): Linear(in_features=2048, out_features=6144, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=2048, out_features=2048, bias=False)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (norm_1): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
        (mlp): MLP(
          (fc1): Linear(in_features=2048, out_features=8192, bias=False)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=8192, out_features=2048, bias=False)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (norm_2): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
      )
      (11): Block(
        (attn): Attention(
          (qkv): Linear(in_features=2048, out_features=6144, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=2048, out_features=2048, bias=False)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (norm_1): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
        (mlp): MLP(
          (fc1): Linear(in_features=2048, out_features=8192, bias=False)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=8192, out_features=2048, bias=False)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (norm_2): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
      )
      (12): Block(
        (attn): Attention(
          (qkv): Linear(in_features=2048, out_features=6144, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=2048, out_features=2048, bias=False)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (norm_1): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
        (mlp): MLP(
          (fc1): Linear(in_features=2048, out_features=8192, bias=False)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=8192, out_features=2048, bias=False)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (norm_2): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
      )
      (13): Block(
        (attn): Attention(
          (qkv): Linear(in_features=2048, out_features=6144, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=2048, out_features=2048, bias=False)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (norm_1): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
        (mlp): MLP(
          (fc1): Linear(in_features=2048, out_features=8192, bias=False)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=8192, out_features=2048, bias=False)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (norm_2): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
      )
      (14): Block(
        (attn): Attention(
          (qkv): Linear(in_features=2048, out_features=6144, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=2048, out_features=2048, bias=False)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (norm_1): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
        (mlp): MLP(
          (fc1): Linear(in_features=2048, out_features=8192, bias=False)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=8192, out_features=2048, bias=False)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (norm_2): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
      )
      (15): Block(
        (attn): Attention(
          (qkv): Linear(in_features=2048, out_features=6144, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=2048, out_features=2048, bias=False)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (norm_1): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
        (mlp): MLP(
          (fc1): Linear(in_features=2048, out_features=8192, bias=False)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=8192, out_features=2048, bias=False)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (norm_2): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
      )
      (16): Block(
        (attn): Attention(
          (qkv): Linear(in_features=2048, out_features=6144, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=2048, out_features=2048, bias=False)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (norm_1): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
        (mlp): MLP(
          (fc1): Linear(in_features=2048, out_features=8192, bias=False)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=8192, out_features=2048, bias=False)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (norm_2): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
      )
      (17): Block(
        (attn): Attention(
          (qkv): Linear(in_features=2048, out_features=6144, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=2048, out_features=2048, bias=False)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (norm_1): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
        (mlp): MLP(
          (fc1): Linear(in_features=2048, out_features=8192, bias=False)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=8192, out_features=2048, bias=False)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (norm_2): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
      )
      (18): Block(
        (attn): Attention(
          (qkv): Linear(in_features=2048, out_features=6144, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=2048, out_features=2048, bias=False)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (norm_1): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
        (mlp): MLP(
          (fc1): Linear(in_features=2048, out_features=8192, bias=False)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=8192, out_features=2048, bias=False)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (norm_2): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
      )
      (19): Block(
        (attn): Attention(
          (qkv): Linear(in_features=2048, out_features=6144, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=2048, out_features=2048, bias=False)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (norm_1): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
        (mlp): MLP(
          (fc1): Linear(in_features=2048, out_features=8192, bias=False)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=8192, out_features=2048, bias=False)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (norm_2): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
      )
      (20): Block(
        (attn): Attention(
          (qkv): Linear(in_features=2048, out_features=6144, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=2048, out_features=2048, bias=False)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (norm_1): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
        (mlp): MLP(
          (fc1): Linear(in_features=2048, out_features=8192, bias=False)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=8192, out_features=2048, bias=False)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (norm_2): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
      )
      (21): Block(
        (attn): Attention(
          (qkv): Linear(in_features=2048, out_features=6144, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=2048, out_features=2048, bias=False)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (norm_1): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
        (mlp): MLP(
          (fc1): Linear(in_features=2048, out_features=8192, bias=False)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=8192, out_features=2048, bias=False)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (norm_2): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
      )
      (22): Block(
        (attn): Attention(
          (qkv): Linear(in_features=2048, out_features=6144, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=2048, out_features=2048, bias=False)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (norm_1): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
        (mlp): MLP(
          (fc1): Linear(in_features=2048, out_features=8192, bias=False)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=8192, out_features=2048, bias=False)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (norm_2): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
      )
      (23): Block(
        (attn): Attention(
          (qkv): Linear(in_features=2048, out_features=6144, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=2048, out_features=2048, bias=False)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (norm_1): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
        (mlp): MLP(
          (fc1): Linear(in_features=2048, out_features=8192, bias=False)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=8192, out_features=2048, bias=False)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (norm_2): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
      )
    )
    (post_trunk_norm): LayerNorm((2048,), eps=1e-06, elementwise_affine=True)
    (post_transformer_layer): AverageLayers()
  )
  (head): AttentionPoolingClassifier(
    (k): Linear(in_features=2048, out_features=2048, bias=False)
    (v): Linear(in_features=2048, out_features=2048, bias=False)
    (linear): Linear(in_features=2048, out_features=1000, bias=True)
    (bn): BatchNorm1d(2048, eps=1e-06, momentum=0.1, affine=False, track_running_stats=True)
  )
)