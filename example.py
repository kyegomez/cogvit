import torch
from cogvit import CogViT, CogViTConfig

cfg = CogViTConfig(
    image_size=224,
    patch_size=14,
    embed_dim=384,
    depth=4,
    num_heads=6,
)

vit = CogViT(cfg)

images = torch.randn(2, 3, 224, 224)
out = vit(images)

print(out["patch_tokens"].shape)
print(out["cls_token"].shape)
print(out["grid"])
