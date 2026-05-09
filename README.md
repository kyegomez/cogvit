## CogViT — Pytorch

Implementation of <a href="https://arxiv.org/abs/2604.26752">CogViT</a>, the parameter-efficient vision encoder from <em>GLM-5V-Turbo: Toward a Native Foundation Model for Multimodal Agents</em>, in Pytorch. A single-file, self-contained reference build of the §2.1 vision tower, the MLP adapter that bridges into the language backbone, and both pretraining stages (DMIM distillation and SigLIP contrastive).

The encoder is a fairly standard <a href="https://arxiv.org/abs/2010.11929">ViT</a> spine with three modern touches: <a href="https://arxiv.org/abs/2010.04245">QK-Norm</a> for stable attention at scale, learned 2D position embeddings that are bicubically interpolated to arbitrary patch grids (the <a href="https://arxiv.org/abs/2502.14786">NaFlex</a> path), and a learnable mask token that lets the same encoder serve both stage-1 masked image modeling and stage-2 contrastive pretraining without architectural surgery.

## Install

```bash
$ pip install -e .
```

## Usage

The encoder by itself:

```python
import torch
from cogvit import CogViT, CogViTConfig

cfg = CogViTConfig(
    image_size = 224,
    patch_size = 14,
    embed_dim  = 1024,
    depth      = 24,
    num_heads  = 16,
)

vit = CogViT(cfg)

images = torch.randn(2, 3, 224, 224)
out = vit(images)

out["patch_tokens"]   # (2, 256, 1024) — Hp*Wp patch tokens
out["cls_token"]      # (2, 1024)
out["grid"]           # (16, 16)
```

## Variable resolution (NaFlex)

Images of arbitrary aspect ratio can be packed into a single batch under a fixed token budget. The encoder consumes a per-patch `valid_mask` so attention ignores padding patches.

```python
from cogvit import CogViT, CogViTConfig, naflex_collate

cfg = CogViTConfig(naflex_max_tokens = 1024)
vit = CogViT(cfg)

images = [torch.randn(3, 480, 640), torch.randn(3, 720, 480)]
batch, valid_mask = naflex_collate(images, patch_size=cfg.patch_size, max_tokens=1024)

out = vit(batch, valid_mask=valid_mask)
```

## Bridging into a language model

`MLPAdapter` projects vision tokens into the LLM's hidden size, with an optional 2x2 pixel-shuffle that quarters the visual sequence length (this is the regime expected by the MMTP head in Figure 2 of the paper).

```python
from cogvit import MLPAdapter, insert_visual_tokens_mmtp

adapter = MLPAdapter(
    vision_dim   = 1024,
    llm_dim      = 4096,
    spatial_pool = 2,    # 16x16 -> 8x8 = 64 visual tokens
)

visual_embeds = adapter(out["patch_tokens"], out["grid"])  # (B, 64, 4096)

# splice into the LLM input stream at <|image|> placeholder positions
fused = insert_visual_tokens_mmtp(
    text_embeds    = text_embeds,        # (B, T, 4096)
    visual_embeds  = visual_embeds,      # (B, 64, 4096)
    image_token_id = IMAGE_TOKEN_ID,
    input_ids      = input_ids,          # (B, T)
)
```

## Stage 1 — Distillation-Based Masked Image Modeling

35% Bernoulli masking on a 224x224 grid. The student reconstructs masked positions in the feature space of two frozen teachers — <a href="https://arxiv.org/abs/2502.14786">SigLIP2</a> for semantics, <a href="https://arxiv.org/abs/2508.10104">DINOv3</a> for texture — under a Smooth-L1 loss.

```python
from cogvit import CogViT, CogViTConfig, CogViTForMIM, TeacherWrapper

cfg = CogViTConfig()
student = CogViT(cfg)

mim = CogViTForMIM(
    encoder = student,
    teachers = {
        "siglip2": TeacherWrapper(siglip2_model, feature_dim=1152, out_grid=(16, 16)),
        "dinov3":  TeacherWrapper(dinov3_model,  feature_dim=1024, out_grid=(16, 16)),
    },
    mask_ratio = 0.35,
)

images = torch.randn(8, 3, 224, 224)
out = mim(images)

out["total"].backward()
```

## Stage 2 — SigLIP contrastive image-text pretraining

Bidirectional all-gather across DDP ranks, learnable temperature and bias, mean-pooling over valid patches.

```python
from cogvit import CogViT, CogViTConfig, CogViTForContrastive, SigLIPLoss

vision = CogViT(CogViTConfig())

clip = CogViTForContrastive(
    vision       = vision,
    text_encoder = my_text_encoder,
    text_dim     = 1024,
    proj_dim     = 768,
    loss         = SigLIPLoss(init_t=10.0, init_b=-10.0),
)

out = clip(
    images         = images,
    input_ids      = input_ids,
    attention_mask = attention_mask,
    valid_mask     = valid_mask,
)

out["loss"].backward()
```

## Optimizer parameter groups

The §2.1 stage-2 recipe assigns module-specific learning rates and decay schedules. `build_param_groups` factors out the bookkeeping — no-decay positions (biases, norms, learned tokens, `log_t`) are routed into a separate group automatically.

```python
from cogvit import build_param_groups

param_groups = build_param_groups(
    ("vision",     clip.vision,       1.0),
    ("text",       clip.text_encoder, 0.5),
    ("vision_proj", clip.vision_proj, 1.0),
    ("text_proj",   clip.text_proj,   1.0),
    weight_decay = 0.05,
)

optim = torch.optim.AdamW(param_groups, lr=1e-3)
# multiply group["lr"] by group["lr_mult"] inside your scheduler step
```

The Muon optimizer cited by the paper is not bundled here; the parameter-group helper is set up so a Muon implementation (or AdamW as a fallback) can be plugged in externally.

## How the model works

A single forward pass through the encoder:

1. **Patch embedding** — a strided `Conv2d` chops the image into non-overlapping `patch_size`-pixel patches and projects each to a `D`-dim token. Inputs whose H/W are multiples of `patch_size` are accepted directly; NaFlex preprocessing pads to the nearest multiple.
2. **Mask token replacement** *(stage-1 only)* — patch positions selected by the Bernoulli mask are overwritten with a single learned mask token *before* position embedding, so the encoder cannot peek at the masked content.
3. **2D position embedding** — a learned `(D, H_base, W_base)` tensor is bicubically interpolated to the actual `(Hp, Wp)` grid and broadcast-added to the tokens. The same parameter therefore serves every input resolution.
4. **CLS token** — optional learned token prepended to the sequence.
5. **L pre-LN transformer blocks** — each block is `x + DropPath(Attn(LN(x)))` followed by `x + DropPath(MLP(LN(x)))`. Attention is multi-head with **QK-Norm**: `Q` and `K` are LayerNorm'd along the per-head feature axis before the dot product, which keeps logits well-conditioned at scale. Optional `LayerScale` and stochastic depth are linearly distributed across blocks.
6. **Final LayerNorm** is applied to the full sequence; the CLS token is split off from the patch tokens for downstream consumers.

The `MLPAdapter` then drops the CLS, optionally pixel-shuffles a `2x2` neighbourhood to quarter the sequence length, and runs a two-layer GELU MLP into the LLM's hidden size. The `insert_visual_tokens_mmtp` helper splices the resulting embeddings into the LLM input stream at `<|image|>` placeholder positions — leaving the placeholder ids intact in `input_ids` so the multi-token-prediction head can still consume them as the shared learnable image token (Figure 2, Option 3).

## Citations

```bibtex
@misc{glm5vturbo2026,
    title   = {GLM-5V-Turbo: Toward a Native Foundation Model for Multimodal Agents},
    author  = {{GLM-5V-Turbo Team}},
    year    = {2026},
    eprint  = {2604.26752},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CV}
}
```

```bibtex
@misc{henry2020querykey,
    title   = {Query-Key Normalization for Transformers},
    author  = {Alex Henry and Prudhvi Raj Dachapally and Shubham Pawar and Yuxuan Chen},
    year    = {2020},
    eprint  = {2010.04245},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CL}
}
```

```bibtex
@misc{zhai2023sigmoid,
    title   = {Sigmoid Loss for Language Image Pre-Training},
    author  = {Xiaohua Zhai and Basil Mustafa and Alexander Kolesnikov and Lucas Beyer},
    year    = {2023},
    eprint  = {2303.15343},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CV}
}
```

```bibtex
@misc{tschannen2025siglip2,
    title   = {SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features},
    author  = {Michael Tschannen and others},
    year    = {2025},
    eprint  = {2502.14786},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CV}
}
```

```bibtex
@misc{simeoni2025dinov3,
    title   = {DINOv3},
    author  = {Oriane Simeoni and others},
    year    = {2025},
    eprint  = {2508.10104},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CV}
}
```

```bibtex
@inproceedings{huang2016deep,
    title     = {Deep Networks with Stochastic Depth},
    author    = {Gao Huang and Yu Sun and Zhuang Liu and Daniel Sedra and Kilian Q. Weinberger},
    booktitle = {ECCV},
    year      = {2016}
}
```

```bibtex
@inproceedings{touvron2021cait,
    title     = {Going Deeper with Image Transformers},
    author    = {Hugo Touvron and Matthieu Cord and Alexandre Sablayrolles and Gabriel Synnaeve and Herv\'{e} J\'{e}gou},
    booktitle = {ICCV},
    year      = {2021}
}
```

## License

Apache License 2.0. See [LICENSE](./LICENSE).
