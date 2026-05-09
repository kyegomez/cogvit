"""
CogViT — Parameter-Efficient Vision Encoder for Multimodal Foundation Models
============================================================================

Reference
---------
GLM-5V-Turbo Team. *GLM-5V-Turbo: Toward a Native Foundation Model for
Multimodal Agents*. arXiv:2604.26752v2 [cs.CV] 6 May 2026.

    §2.1  CogViT Vision Encoder
    §2.2  Multimodal Multi-Token Prediction (MMTP)
    Fig.2 CogViT + MLP Adapter feeding the LLM backbone

This single file is a self-contained PyTorch reference implementation of
CogViT and the surrounding training apparatus described in §2.1. It
covers:

    1. The CogViT architecture
       - patch embedding with NaFlex variable-resolution support
       - 2D learned position embeddings interpolated to arbitrary grids
       - L-layer Transformer with QK-Norm self-attention
       - CLS token + final layer norm
    2. The MLP adapter that bridges CogViT tokens to a language model
       hidden size, with optional 2x2 spatial pooling.
    3. The MMTP <|image|> placeholder utility used by GLM-5V-Turbo's
       multi-token-prediction head (Fig. 2 — Option 3, "adopted").
    4. Stage-1 head — ``CogViTForMIM``: distillation-based masked image
       modeling against dual frozen teachers (SigLIP2 for semantic
       features, DINOv3 for texture features). 35% masking @ 224x224.
    5. Stage-2 head — ``CogViTForContrastive``: image-text pretraining
       with a bidirectional sigmoid (SigLIP) loss suitable for
       large-batch distributed training.

The Muon optimizer cited by the paper is not bundled here; the file
exposes parameter-group helpers so a Muon implementation (or AdamW as a
fallback) can be plugged in externally.


Algorithmic Pseudocode
----------------------

ARCH — single forward pass of CogViT::

    input  : x of shape (B, 3, H, W),  optional mask M of shape (B, N)
    output : tokens of shape (B, N (+1 if cls), D)

    Hp, Wp  <- H / patch_size, W / patch_size
    N       <- Hp * Wp
    tokens  <- Conv2d(patch) (x).flatten(2).transpose(1, 2)          # (B, N, D)
    if M given: tokens[M] <- mask_token                              # stage-1 only
    tokens  <- tokens + interp_2d_pos_embed(grid=(Hp, Wp))
    if cls: tokens <- concat([cls_token, tokens], dim=1)             # N <- N + 1
    for block in transformer_blocks (L blocks):
        h       <- LayerNorm(tokens)
        q, k, v <- linear_qkv(h).chunk(3)
        q       <- LayerNorm(q)            # QK-Norm  (Henry et al. 2020)
        k       <- LayerNorm(k)            # QK-Norm
        a       <- scaled_dot_product_attention(q, k, v, attn_mask)
        tokens  <- tokens + drop_path(linear_o(a))
        h       <- LayerNorm(tokens)
        tokens  <- tokens + drop_path(MLP(h))                        # GELU 4x
    return final_norm(tokens)


ADAPTER — CogViT tokens -> LLM hidden::

    drop CLS, optionally pixel-shuffle 2x2 to halve sequence length,
    Linear(D -> Dh) -> GELU -> Linear(Dh -> D_llm).
    The result is fed into the LLM at <|image|> placeholder positions.


STAGE 1 — Distillation-Based Masked Image Modeling (DMIM)::

    for image x in dataset:
        M           <- bernoulli_mask(N, ratio=0.35)
        student     <- CogViT(x, mask=M)                             # (B, N, D)
        f_siglip    <- SigLIP2_teacher(x).detach()                   # (B, N, Ds)
        f_dinov3    <- DINOv3_teacher(x).detach()                    # (B, N, Dd)
        pred_s      <- proj_s(student)
        pred_d      <- proj_d(student)
        loss        <- smooth_l1(pred_s[M], f_siglip[M])
                     + smooth_l1(pred_d[M], f_dinov3[M])
        optimize loss with Muon  +  cosine LR decay


STAGE 2 — SigLIP Contrastive Image-Text Pretraining::

    for (x_i, y_i) batch (B local, B_global = 64K via DDP):
        v_i  <- mean_pool(CogViT(naflex(x_i))) -> proj_v -> l2norm
        t_i  <- TextEncoder(y_i)               -> proj_t -> l2norm
        v_all, t_all <- bidirectional_all_gather(v_i, t_i)
        logits <- temperature * v_all @ t_all^T + bias
        labels <- 2 * I_{B_global} - 1                               # +1 diag, -1 off
        loss   <- -mean( log sigmoid(labels * logits) )
        optimize loss with Muon, module-specific LR & decay
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class CogViTConfig:
    """Hyperparameter configuration for the CogViT vision encoder.

    Defaults correspond to a ~1B-parameter vision tower comparable to the
    one described in §2.1 (parameter-efficient, two-stage pretraining).
    Adjust ``embed_dim``, ``depth`` and ``num_heads`` to scale.

    Attributes:
        image_size: Square edge length of the *base* image used for stage-1
            DMIM and for shaping the learned 2D position embedding. Inputs
            at inference time may have any (H, W) divisible by ``patch_size``;
            position embeddings are bicubically interpolated to fit.
        patch_size: Side length (in pixels) of each non-overlapping patch.
        in_chans: Number of input image channels (3 for RGB).
        embed_dim: Token feature dimension ``D`` carried through every block.
        depth: Number ``L`` of stacked transformer blocks.
        num_heads: Number of self-attention heads. Must divide ``embed_dim``.
        mlp_ratio: Hidden-to-input ratio of the two-layer MLP in each block.
        qkv_bias: Whether the QKV projection uses a bias term.
        proj_bias: Whether the post-attention output projection uses a bias.
        use_qk_norm: Whether to apply LayerNorm to Q and K before the
            attention dot product (Henry et al. 2020). Strongly recommended
            at scale per §2.1.
        use_cls_token: Whether to prepend a learned ``[CLS]`` token. Stage-2
            mean-pools patch tokens, so the CLS token is optional there.
        drop_path_rate: Maximum stochastic-depth rate. Linearly distributed
            from ``0`` at block 0 to ``drop_path_rate`` at the last block.
        layer_scale_init: If non-``None``, residual branches are scaled by a
            learnable per-channel parameter initialized to this value.
        naflex_max_tokens: NaFlex token-budget cap used by
            :func:`naflex_resize` / :func:`naflex_collate` when batching
            variable-resolution images.
        eps: ``LayerNorm`` epsilon used throughout the encoder.
    """

    image_size: int = 224
    patch_size: int = 14
    in_chans: int = 3

    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0

    qkv_bias: bool = True
    proj_bias: bool = True
    use_qk_norm: bool = True
    use_cls_token: bool = True

    drop_path_rate: float = 0.0
    layer_scale_init: Optional[float] = None  # e.g. 1e-5 to enable LayerScale

    naflex_max_tokens: int = 1024
    eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        """Per-head feature dim. Raises if ``embed_dim`` is not divisible by ``num_heads``."""
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        return self.embed_dim // self.num_heads

    @property
    def base_grid(self) -> Tuple[int, int]:
        """Base patch grid ``(Hp, Wp)`` produced from a square ``image_size`` image."""
        g = self.image_size // self.patch_size
        return g, g


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """Conv2d patch embedding.

    Accepts inputs whose spatial dimensions are multiples of ``patch_size``
    (NaFlex preprocessing pads images to the nearest multiple). Returns a
    flattened token sequence and the resulting (Hp, Wp) grid.

    Args:
        cfg: Encoder configuration; uses ``in_chans``, ``embed_dim`` and
            ``patch_size``.
    """

    patch_size: int
    proj: nn.Conv2d

    def __init__(self, cfg: CogViTConfig) -> None:
        super().__init__()
        self.patch_size = cfg.patch_size
        self.proj = nn.Conv2d(
            cfg.in_chans,
            cfg.embed_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        """Embed a batch of images into a flat patch-token sequence.

        Args:
            x: ``(B, C, H, W)`` image tensor. ``H`` and ``W`` must each be
                a multiple of ``patch_size``.

        Returns:
            tokens: ``(B, Hp*Wp, D)`` patch tokens.
            grid: ``(Hp, Wp)`` patch grid where ``Hp = H // patch_size``.
        """
        if x.shape[-1] % self.patch_size or x.shape[-2] % self.patch_size:
            raise ValueError(
                f"Input ({x.shape[-2]}x{x.shape[-1]}) must be divisible by patch_size={self.patch_size}"
            )
        feat = self.proj(x)
        Hp, Wp = int(feat.shape[-2]), int(feat.shape[-1])
        feat = feat.flatten(2).transpose(1, 2)
        return feat, (Hp, Wp)


class TwoDPositionalEmbedding(nn.Module):
    """Learned 2D positional embedding with bicubic interpolation to arbitrary grids.

    A base grid of ``cfg.base_grid`` parameters is stored; on each forward
    the embedding is resized to the grid actually produced by the patch
    embedder. This is the simple pre-NaFlex approach used as a fallback
    for variable-resolution inputs (§2.1, stage-2 NaFlex scheme).

    Args:
        cfg: Encoder configuration.
    """

    embed: nn.Parameter
    base_h: int
    base_w: int

    def __init__(self, cfg: CogViTConfig) -> None:
        super().__init__()
        h, w = cfg.base_grid
        self.base_h = h
        self.base_w = w
        self.embed = nn.Parameter(torch.zeros(1, cfg.embed_dim, h, w))
        nn.init.trunc_normal_(self.embed, std=0.02)

    def forward(self, grid: Tuple[int, int]) -> Tensor:
        """Return the position embedding evaluated on a target patch grid.

        Args:
            grid: Target ``(Hp, Wp)`` patch grid.

        Returns:
            ``(1, Hp*Wp, D)`` positional embedding ready to broadcast-add to
            patch tokens.
        """
        Hp, Wp = grid
        if (Hp, Wp) == (self.base_h, self.base_w):
            pos = self.embed
        else:
            pos = F.interpolate(
                self.embed,
                size=(Hp, Wp),
                mode="bicubic",
                align_corners=False,
            )
        return pos.flatten(2).transpose(1, 2)


class QKNormAttention(nn.Module):
    """Multi-head self-attention with QK-Norm.

    Implements the QK-Norm scheme of Henry et al. (2020) cited by §2.1: query
    and key vectors are LayerNorm'd before the dot product, mitigating logit
    explosion at scale. Uses ``F.scaled_dot_product_attention`` so the
    runtime can dispatch to flash-attention on supported hardware.
    """

    num_heads: int
    head_dim: int
    scale: float
    qkv: nn.Linear
    proj: nn.Linear
    q_norm: nn.Module
    k_norm: nn.Module

    def __init__(self, cfg: CogViTConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(cfg.embed_dim, cfg.embed_dim * 3, bias=cfg.qkv_bias)
        self.proj = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=cfg.proj_bias)

        if cfg.use_qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim, eps=cfg.eps)
            self.k_norm = nn.LayerNorm(self.head_dim, eps=cfg.eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """Run multi-head self-attention with QK-Norm.

        Args:
            x: ``(B, N, D)`` input token sequence.
            attn_mask: Optional bool tensor broadcastable to
                ``(B, num_heads, N, N)``. ``True`` entries keep, ``False``
                entries are masked out (used to suppress attention over
                NaFlex padding patches).

        Returns:
            ``(B, N, D)`` attended-and-projected output.
        """
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = self.q_norm(q)
        k = self.k_norm(k)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, scale=self.scale
        )
        out = out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        return self.proj(out)


class FeedForward(nn.Module):
    """Two-layer MLP with GELU. Hidden dim = ``int(dim * mlp_ratio)``.

    Args:
        dim: Input/output feature dimension.
        mlp_ratio: Ratio of hidden dim to ``dim``.
    """

    fc1: nn.Linear
    fc2: nn.Linear
    act: nn.Module

    def __init__(self, dim: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        """Apply ``fc1 -> GELU -> fc2`` to a ``(*, dim)`` tensor."""
        return self.fc2(self.act(self.fc1(x)))


class DropPath(nn.Module):
    """Stochastic depth per residual branch (Huang et al. 2016).

    During training, each sample's residual contribution is dropped with
    probability ``drop_prob`` and the rest are rescaled by ``1/(1-p)`` so
    that the expected value is preserved.

    Args:
        drop_prob: Drop probability in ``[0, 1)``.
    """

    drop_prob: float

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError("drop_prob must be in [0, 1)")
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        """Stochastically zero whole rows of ``x`` along the batch axis."""
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask


class LayerScale(nn.Module):
    """Learnable per-channel residual scale (Touvron et al., CaiT 2021).

    Args:
        dim: Channel dimension to scale.
        init: Initial value for every element of ``gamma``.
    """

    gamma: nn.Parameter

    def __init__(self, dim: int, init: float) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init))

    def forward(self, x: Tensor) -> Tensor:
        """Multiply the last dimension of ``x`` by the learned ``gamma``."""
        return x * self.gamma


class TransformerBlock(nn.Module):
    """Pre-LN transformer block with QK-Norm attention and stochastic depth.

    Order of operations matches §2.1 / Fig. 2:

    .. code-block::

        x <- x + DropPath(LS1(Attn(LN(x))))
        x <- x + DropPath(LS2(MLP(LN(x))))

    Args:
        cfg: Encoder configuration.
        drop_path: Stochastic-depth rate for *this* block (linearly scaled
            from ``0`` to ``cfg.drop_path_rate`` across the depth).
    """

    norm1: nn.LayerNorm
    attn: QKNormAttention
    norm2: nn.LayerNorm
    mlp: FeedForward
    drop_path: DropPath
    ls1: nn.Module
    ls2: nn.Module

    def __init__(self, cfg: CogViTConfig, drop_path: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.embed_dim, eps=cfg.eps)
        self.attn = QKNormAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.embed_dim, eps=cfg.eps)
        self.mlp = FeedForward(cfg.embed_dim, cfg.mlp_ratio)
        self.drop_path = DropPath(drop_path)

        if cfg.layer_scale_init is not None:
            self.ls1 = LayerScale(cfg.embed_dim, cfg.layer_scale_init)
            self.ls2 = LayerScale(cfg.embed_dim, cfg.layer_scale_init)
        else:
            self.ls1 = nn.Identity()
            self.ls2 = nn.Identity()

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """Run one pre-LN transformer block.

        Args:
            x: ``(B, N, D)`` token sequence.
            attn_mask: Optional bool mask broadcastable to
                ``(B, num_heads, N, N)``.

        Returns:
            ``(B, N, D)`` updated token sequence.
        """
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x


# ---------------------------------------------------------------------------
# CogViT encoder
# ---------------------------------------------------------------------------


class CogViT(nn.Module):
    """CogViT — the vision encoder of GLM-5V-Turbo (§2.1).

    Args:
        cfg: See :class:`CogViTConfig`.

    Forward:
        x: ``(B, 3, H, W)`` image tensor whose spatial size is divisible by
            ``cfg.patch_size``. Variable-size NaFlex inputs should first be
            preprocessed via :func:`naflex_resize` and batched with the
            ``valid_mask`` returned by :func:`naflex_collate`.
        mask: optional bool tensor of shape ``(B, Hp*Wp)`` marking patch
            tokens to be replaced by the learned mask token (used by
            :class:`CogViTForMIM`).
        valid_mask: optional bool tensor of shape ``(B, Hp*Wp)``; ``False``
            entries are padding tokens whose attention is suppressed.

    Returns dict with:
        ``patch_tokens``: ``(B, Hp*Wp, D)`` post-norm patch features.
        ``cls_token``:    ``(B, D)`` if ``cfg.use_cls_token`` else ``None``.
        ``grid``:         ``(Hp, Wp)`` grid that produced the tokens.
    """

    cfg: CogViTConfig
    patch_embed: PatchEmbed
    pos_embed: TwoDPositionalEmbedding
    blocks: nn.ModuleList
    norm: nn.LayerNorm
    mask_token: nn.Parameter
    cls_token: Optional[nn.Parameter]

    def __init__(self, cfg: CogViTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.patch_embed = PatchEmbed(cfg)
        self.pos_embed = TwoDPositionalEmbedding(cfg)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        if cfg.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.register_parameter("cls_token", None)

        dpr = torch.linspace(0, cfg.drop_path_rate, cfg.depth).tolist()
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, drop_path=dpr[i]) for i in range(cfg.depth)]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim, eps=cfg.eps)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Truncated-normal init for ``Linear``, identity init for ``LayerNorm``."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _build_attn_mask(
        self, valid_mask: Optional[Tensor], has_cls: bool
    ) -> Optional[Tensor]:
        """Promote a per-patch validity mask into an SDPA-compatible attention mask.

        Args:
            valid_mask: ``(B, N)`` bool tensor; ``False`` entries are
                NaFlex padding and should be masked out.
            has_cls: Whether the encoder prepends a CLS token (always
                valid).

        Returns:
            Either ``None`` when no mask is needed, or a ``(B, 1, 1, N')``
            bool tensor where ``N' = N + has_cls``.
        """
        if valid_mask is None:
            return None
        if has_cls:
            cls = valid_mask.new_ones(valid_mask.shape[0], 1, dtype=torch.bool)
            valid = torch.cat([cls, valid_mask], dim=1)
        else:
            valid = valid_mask
        return valid[:, None, None, :]

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        valid_mask: Optional[Tensor] = None,
    ) -> Dict[str, object]:
        """Encode a batch of (possibly NaFlex-padded) images.

        Args:
            x: ``(B, C, H, W)`` image tensor.
            mask: Optional ``(B, Hp*Wp)`` bool mask; ``True`` positions are
                replaced by the learned mask token before position embedding
                (used by stage-1 :class:`CogViTForMIM`).
            valid_mask: Optional ``(B, Hp*Wp)`` bool mask marking real
                (non-padding) patches under NaFlex collation.

        Returns:
            A dict with keys
                ``patch_tokens`` -> ``(B, Hp*Wp, D)`` post-norm patch features,
                ``cls_token``    -> ``(B, D)`` CLS feature or ``None``,
                ``grid``         -> ``(Hp, Wp)`` patch grid.
        """
        tokens, grid = self.patch_embed(x)  # (B, N, D)
        B, N, D = tokens.shape

        if mask is not None:
            if mask.shape != (B, N):
                raise ValueError(f"mask shape {tuple(mask.shape)} != ({B}, {N})")
            tokens = torch.where(
                mask.unsqueeze(-1), self.mask_token.expand(B, N, D), tokens
            )

        tokens = tokens + self.pos_embed(grid)

        has_cls = self.cls_token is not None
        if has_cls:
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

        attn_mask = self._build_attn_mask(valid_mask, has_cls=has_cls)
        for block in self.blocks:
            tokens = block(tokens, attn_mask=attn_mask)

        tokens = self.norm(tokens)

        if has_cls:
            cls_out = tokens[:, 0]
            patch = tokens[:, 1:]
        else:
            cls_out = None
            patch = tokens

        return {"patch_tokens": patch, "cls_token": cls_out, "grid": grid}


# ---------------------------------------------------------------------------
# MLP adapter (CogViT -> LLM)
# ---------------------------------------------------------------------------


class MLPAdapter(nn.Module):
    """Bridge module that projects CogViT patch tokens into the LLM hidden size.

    Optionally applies a ``spatial_pool x spatial_pool`` pixel-shuffle on the
    2D patch grid before projecting. With ``spatial_pool=2`` the visual
    sequence length is reduced 4x, which is the regime expected by the MMTP
    head in Figure 2 of the paper.

    Args:
        vision_dim: ``D`` — channel dim of the patch tokens emitted by
            :class:`CogViT`.
        llm_dim: Hidden size of the downstream language model.
        hidden_dim: Width of the intermediate MLP layer. Defaults to
            ``llm_dim``.
        spatial_pool: Factor for non-overlapping ``s x s`` pixel-shuffle on
            the patch grid. ``1`` disables pooling.
    """

    spatial_pool: int
    fc1: nn.Linear
    fc2: nn.Linear
    act: nn.Module

    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        hidden_dim: Optional[int] = None,
        spatial_pool: int = 1,
    ) -> None:
        super().__init__()
        if spatial_pool < 1:
            raise ValueError("spatial_pool must be >= 1")
        self.spatial_pool = spatial_pool
        in_dim = vision_dim * (spatial_pool**2)
        hidden = hidden_dim or llm_dim
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, llm_dim)

    def forward(self, patch_tokens: Tensor, grid: Tuple[int, int]) -> Tensor:
        """Pixel-shuffle (optional) then 2-layer MLP project to LLM hidden.

        Args:
            patch_tokens: ``(B, Hp*Wp, vision_dim)`` patch features from
                :class:`CogViT`.
            grid: ``(Hp, Wp)`` original patch grid.

        Returns:
            ``(B, (Hp/s) * (Wp/s), llm_dim)`` adapted visual prefix.
        """
        B, N, D = patch_tokens.shape
        Hp, Wp = grid
        if Hp * Wp != N:
            raise ValueError(f"grid {grid} does not match N={N}")

        if self.spatial_pool > 1:
            s = self.spatial_pool
            if Hp % s or Wp % s:
                raise ValueError(f"grid {grid} not divisible by spatial_pool={s}")
            x = patch_tokens.reshape(B, Hp, Wp, D)
            x = x.reshape(B, Hp // s, s, Wp // s, s, D)
            x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, (Hp // s) * (Wp // s), D * s * s)
        else:
            x = patch_tokens

        return self.fc2(self.act(self.fc1(x)))


# ---------------------------------------------------------------------------
# MMTP integration helper
# ---------------------------------------------------------------------------


def insert_visual_tokens_mmtp(
    text_embeds: Tensor,
    visual_embeds: Tensor,
    image_token_id: int,
    input_ids: Tensor,
) -> Tensor:
    """Replace ``<|image|>`` placeholder embeddings with visual embeddings.

    Implements the "Option 3" placement described in §2.2 / Fig. 2: the LLM
    backbone receives true visual embeddings, while the MTP head consumes a
    sequence in which every visual position has been collapsed back to the
    shared learnable ``<|image|>`` token. This helper is for the *backbone*
    path; the MTP head path simply consumes ``text_embeds`` produced from
    ``input_ids`` (where image positions still carry the placeholder token
    id) and is a single embedding lookup.

    Args:
        text_embeds: ``(B, T, D)`` token embeddings looked up from
            ``input_ids``.
        visual_embeds: ``(B, V, D)`` visual embeddings emitted by
            :class:`MLPAdapter`. ``V`` must equal the total number of
            ``image_token_id`` entries in ``input_ids`` for each row.
        image_token_id: token id used for the ``<|image|>`` placeholder.
        input_ids: ``(B, T)`` token ids.

    Returns:
        ``(B, T, D)`` embeddings with placeholder positions overwritten.
    """
    if text_embeds.shape[:2] != input_ids.shape:
        raise ValueError("text_embeds and input_ids must share (B, T)")

    B, T, D = text_embeds.shape
    out = text_embeds.clone()
    placeholder = input_ids == image_token_id
    counts = placeholder.sum(dim=1)
    expected = visual_embeds.shape[1]
    if not torch.all(counts == expected):
        raise ValueError(
            f"each row must have exactly {expected} <|image|> tokens; got {counts.tolist()}"
        )
    out[placeholder] = visual_embeds.reshape(-1, D).to(out.dtype)
    return out


# ---------------------------------------------------------------------------
# NaFlex preprocessing utilities
# ---------------------------------------------------------------------------


def naflex_target_size(
    h: int, w: int, patch_size: int, max_tokens: int
) -> Tuple[int, int]:
    """Compute a NaFlex target ``(H', W')`` that preserves aspect ratio.

    The resulting size satisfies ``(H'/p) * (W'/p) <= max_tokens`` and both
    dimensions are multiples of ``patch_size``.
    """
    if h <= 0 or w <= 0:
        raise ValueError("h and w must be positive")
    aspect = h / w
    # H' * W' <= max_tokens * p^2 ; H' = aspect * W'
    max_area = max_tokens * patch_size * patch_size
    w_star = math.sqrt(max_area / aspect)
    h_star = aspect * w_star
    Hp = max(1, int(h_star // patch_size))
    Wp = max(1, int(w_star // patch_size))
    while Hp * Wp > max_tokens:
        if Hp >= Wp:
            Hp -= 1
        else:
            Wp -= 1
    return Hp * patch_size, Wp * patch_size


def naflex_resize(image: Tensor, patch_size: int, max_tokens: int) -> Tensor:
    """Resize a single ``(C, H, W)`` image to a NaFlex-compatible shape."""
    if image.ndim != 3:
        raise ValueError("expected (C, H, W) input")
    _, h, w = image.shape
    Ht, Wt = naflex_target_size(h, w, patch_size, max_tokens)
    return F.interpolate(
        image.unsqueeze(0).float(), size=(Ht, Wt), mode="bilinear", align_corners=False
    ).squeeze(0)


def naflex_collate(
    images: List[Tensor], patch_size: int, max_tokens: int
) -> Tuple[Tensor, Tensor]:
    """Pad a list of NaFlex-resized images into a single batched tensor.

    Returns the padded ``(B, C, H_max, W_max)`` tensor and a per-patch
    ``valid_mask`` of shape ``(B, (H_max/p) * (W_max/p))`` that the encoder
    consumes to suppress attention over padding patches.
    """
    if not images:
        raise ValueError("images list is empty")
    resized = [naflex_resize(im, patch_size, max_tokens) for im in images]
    H_max = max(im.shape[1] for im in resized)
    W_max = max(im.shape[2] for im in resized)
    Hp_max, Wp_max = H_max // patch_size, W_max // patch_size

    B = len(resized)
    C = resized[0].shape[0]
    out = resized[0].new_zeros(B, C, H_max, W_max)
    mask = torch.zeros(B, Hp_max * Wp_max, dtype=torch.bool)
    for i, im in enumerate(resized):
        c, h, w = im.shape
        out[i, :, :h, :w] = im
        Hp, Wp = h // patch_size, w // patch_size
        m = mask[i].view(Hp_max, Wp_max)
        m[:Hp, :Wp] = True
    return out, mask


# ---------------------------------------------------------------------------
# Stage 1: Distillation-Based Masked Image Modeling
# ---------------------------------------------------------------------------


class TeacherWrapper(nn.Module):
    """Thin wrapper around a frozen teacher model.

    The teacher is expected to return per-patch features of shape
    ``(B, N_t, D_t)`` aligned with a regular grid; if ``out_grid`` differs
    from the student grid, features are interpolated bicubically so that
    the per-patch reconstruction loss is well defined.

    Args:
        model: Frozen feature extractor returning ``(B, N_t, D_t)`` patch
            features for a given image batch.
        feature_dim: ``D_t`` — channel size of the teacher features.
        out_grid: ``(sH, sW)`` patch grid emitted by the teacher. If
            ``None`` the teacher is assumed to already produce features on
            the student grid.
    """

    model: nn.Module
    feature_dim: int
    out_grid: Optional[Tuple[int, int]]

    def __init__(
        self,
        model: nn.Module,
        feature_dim: int,
        out_grid: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.feature_dim = feature_dim
        self.out_grid = out_grid
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def forward(self, x: Tensor, target_grid: Tuple[int, int]) -> Tensor:
        """Run the frozen teacher and resample features to ``target_grid``.

        Args:
            x: ``(B, C, H, W)`` image batch.
            target_grid: ``(Hp, Wp)`` student patch grid.

        Returns:
            ``(B, Hp*Wp, feature_dim)`` teacher features aligned with the
            student grid.
        """
        feats = self.model(x)
        if feats.ndim != 3:
            raise ValueError("teacher must emit (B, N, D) features")
        Hp, Wp = target_grid
        if self.out_grid is None or self.out_grid == (Hp, Wp):
            return feats
        sH, sW = self.out_grid
        if feats.shape[1] != sH * sW:
            raise ValueError(
                f"teacher feature N={feats.shape[1]} != {sH}*{sW} for out_grid={self.out_grid}"
            )
        f = feats.transpose(1, 2).reshape(feats.shape[0], -1, sH, sW)
        f = F.interpolate(f, size=(Hp, Wp), mode="bicubic", align_corners=False)
        return f.flatten(2).transpose(1, 2)


class CogViTForMIM(nn.Module):
    """Stage-1 wrapper: distillation-based masked image modeling.

    Reproduces the §2.1 stage-1 recipe: random Bernoulli masking with
    ``mask_ratio`` (default 0.35) on a fixed 224x224 grid, dual frozen
    teachers (SigLIP2 for semantic features, DINOv3 for texture features),
    Smooth-L1 reconstruction loss in each teacher's feature space.

    The teachers are pluggable :class:`TeacherWrapper` instances so that
    arbitrary semantic and texture teachers can be supplied.

    Args:
        encoder: Student :class:`CogViT` encoder being trained.
        teachers: Mapping from a name (e.g. ``"siglip2"``, ``"dinov3"``)
            to a frozen :class:`TeacherWrapper`.
        mask_ratio: Fraction of patch tokens replaced by the mask token
            on each forward; must lie in ``(0, 1)``.
        loss_weights: Optional per-teacher scalar weights; defaults to
            ``1.0`` for every teacher.
    """

    encoder: "CogViT"
    teachers: nn.ModuleDict
    heads: nn.ModuleDict
    mask_ratio: float
    loss_weights: Dict[str, float]

    def __init__(
        self,
        encoder: CogViT,
        teachers: Dict[str, TeacherWrapper],
        mask_ratio: float = 0.35,
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        if not (0.0 < mask_ratio < 1.0):
            raise ValueError("mask_ratio must be in (0, 1)")
        if not teachers:
            raise ValueError("at least one teacher is required")

        self.encoder = encoder
        self.teachers = nn.ModuleDict(teachers)
        self.mask_ratio = mask_ratio
        self.loss_weights = loss_weights or {k: 1.0 for k in teachers}

        D = encoder.cfg.embed_dim
        self.heads = nn.ModuleDict(
            {name: nn.Linear(D, t.feature_dim) for name, t in teachers.items()}
        )

    @staticmethod
    def random_mask(B: int, N: int, ratio: float, device: torch.device) -> Tensor:
        """Sample a per-row binary mask with ``round(N * ratio)`` masked positions.

        Args:
            B: Batch size.
            N: Number of patches per row.
            ratio: Fraction in ``(0, 1)`` of positions to mask.
            device: Device for the sampled mask.

        Returns:
            ``(B, N)`` bool tensor; ``True`` positions are masked.
        """
        n_mask = max(1, int(round(N * ratio)))
        noise = torch.rand(B, N, device=device)
        thresh = torch.kthvalue(noise, n_mask, dim=1, keepdim=True).values
        return noise <= thresh

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Run one stage-1 step and return per-teacher and total losses.

        Args:
            x: ``(B, 3, image_size, image_size)`` image batch.

        Returns:
            A dict containing one entry per teacher (each a scalar loss),
            a ``"total"`` weighted-sum scalar, and the sampled ``"mask"``.
        """
        cfg = self.encoder.cfg
        if x.shape[-1] != cfg.image_size or x.shape[-2] != cfg.image_size:
            raise ValueError(
                f"stage-1 expects {cfg.image_size}x{cfg.image_size} inputs"
            )
        B = x.shape[0]
        Hp, Wp = cfg.base_grid
        N = Hp * Wp

        mask = self.random_mask(B, N, self.mask_ratio, x.device)
        student_out = self.encoder(x, mask=mask)
        student_patches = student_out["patch_tokens"]

        losses = {}
        total = student_patches.new_zeros(())
        for name, teacher in self.teachers.items():
            target = teacher(x, target_grid=(Hp, Wp))
            pred = self.heads[name](student_patches)
            loss = F.smooth_l1_loss(pred[mask], target[mask])
            w = self.loss_weights.get(name, 1.0)
            losses[name] = loss
            total = total + w * loss

        losses["total"] = total
        return {**losses, "mask": mask}


# ---------------------------------------------------------------------------
# Stage 2: SigLIP contrastive image-text pretraining
# ---------------------------------------------------------------------------


def _all_gather_with_grad(x: Tensor) -> Tensor:
    """All-gather a tensor across DDP ranks with a straight-through gradient.

    Falls back to identity when distributed is unavailable / uninitialized.
    The local rank's slice carries gradients; remote slices are detached
    (this matches the bidirectional all-gather scheme used in stage 2).

    Args:
        x: ``(B, ...)`` tensor on every rank.

    Returns:
        ``(B * world_size, ...)`` concatenation across ranks (or ``x``
        unchanged when running on a single rank).
    """
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return x
    world = torch.distributed.get_world_size()
    if world == 1:
        return x
    rank = torch.distributed.get_rank()
    gathered = [torch.zeros_like(x) for _ in range(world)]
    torch.distributed.all_gather(gathered, x.detach())
    gathered[rank] = x
    return torch.cat(gathered, dim=0)


class SigLIPLoss(nn.Module):
    """Sigmoid pairwise contrastive loss.

    Implements the SigLIP objective with a learnable temperature ``t`` and
    bias ``b`` (Zhai et al. 2023; SigLIP2 — Tschannen et al. 2025). Uses a
    bidirectional all-gather so that both the row and column views of the
    similarity matrix carry gradients into the local rank.

    Args:
        init_t: Initial temperature scale ``t``. Stored internally as
            ``log_t`` for stability.
        init_b: Initial value of the additive logit bias.
    """

    log_t: nn.Parameter
    bias: nn.Parameter

    def __init__(self, init_t: float = 10.0, init_b: float = -10.0) -> None:
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(math.log(init_t)))
        self.bias = nn.Parameter(torch.tensor(float(init_b)))

    def forward(self, image_embed: Tensor, text_embed: Tensor) -> Tensor:
        """Compute the SigLIP loss over a (possibly DDP-gathered) batch.

        Args:
            image_embed: ``(B, D)`` per-rank image embeddings.
            text_embed: ``(B, D)`` per-rank text embeddings paired
                row-wise with ``image_embed``.

        Returns:
            Scalar loss.
        """
        if image_embed.shape != text_embed.shape:
            raise ValueError("image and text embeddings must share shape")

        v = F.normalize(image_embed, dim=-1)
        t = F.normalize(text_embed, dim=-1)

        v_all = _all_gather_with_grad(v)
        t_all = _all_gather_with_grad(t)

        logits = self.log_t.exp() * v_all @ t_all.t() + self.bias
        n = logits.shape[0]
        labels = 2.0 * torch.eye(n, device=logits.device, dtype=logits.dtype) - 1.0
        return -F.logsigmoid(labels * logits).mean()


class CogViTForContrastive(nn.Module):
    """Stage-2 wrapper: NaFlex image-text contrastive pretraining.

    Args:
        vision: Trained-or-being-trained :class:`CogViT` encoder.
        text_encoder: Any module mapping ``(input_ids, attention_mask)``
            to ``(B, text_dim)`` pooled embeddings.
        text_dim: ``D_t`` — output dim of ``text_encoder``.
        proj_dim: Shared embedding dimension for the contrastive space.
        loss: Optional pre-built :class:`SigLIPLoss`. If ``None`` a fresh
            one is constructed with default initial temperature/bias.
    """

    vision: "CogViT"
    text_encoder: nn.Module
    vision_proj: nn.Linear
    text_proj: nn.Linear
    loss_fn: SigLIPLoss

    def __init__(
        self,
        vision: CogViT,
        text_encoder: nn.Module,
        text_dim: int,
        proj_dim: int = 768,
        loss: Optional[SigLIPLoss] = None,
    ) -> None:
        super().__init__()
        self.vision = vision
        self.text_encoder = text_encoder
        self.vision_proj = nn.Linear(vision.cfg.embed_dim, proj_dim, bias=False)
        self.text_proj = nn.Linear(text_dim, proj_dim, bias=False)
        self.loss_fn = loss or SigLIPLoss()

    def encode_image(self, x: Tensor, valid_mask: Optional[Tensor] = None) -> Tensor:
        """Encode images, mean-pool over valid patches, and project.

        Args:
            x: ``(B, C, H, W)`` (NaFlex-padded) image batch.
            valid_mask: Optional ``(B, Hp*Wp)`` bool mask flagging real
                patches; padding patches are excluded from the mean.

        Returns:
            ``(B, proj_dim)`` projected image embedding.
        """
        out = self.vision(x, valid_mask=valid_mask)
        patch = out["patch_tokens"]
        if valid_mask is not None:
            denom = valid_mask.sum(dim=1, keepdim=True).clamp_min(1).to(patch.dtype)
            pooled = (patch * valid_mask.unsqueeze(-1).to(patch.dtype)).sum(
                dim=1
            ) / denom
        else:
            pooled = patch.mean(dim=1)
        return self.vision_proj(pooled)

    def encode_text(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Encode tokenized text and project into the contrastive space.

        Args:
            input_ids: ``(B, T)`` token ids.
            attention_mask: ``(B, T)`` attention mask consumed by
                ``text_encoder``.

        Returns:
            ``(B, proj_dim)`` projected text embedding.
        """
        feats = self.text_encoder(input_ids, attention_mask)
        return self.text_proj(feats)

    def forward(
        self,
        images: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        valid_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Run one stage-2 step and return loss + projected embeddings.

        Args:
            images: ``(B, C, H, W)`` image batch.
            input_ids: ``(B, T)`` paired token ids.
            attention_mask: ``(B, T)`` text attention mask.
            valid_mask: Optional ``(B, Hp*Wp)`` NaFlex validity mask.

        Returns:
            Dict with ``"loss"`` (scalar), ``"image_embed"`` and
            ``"text_embed"``.
        """
        v = self.encode_image(images, valid_mask=valid_mask)
        t = self.encode_text(input_ids, attention_mask)
        loss = self.loss_fn(v, t)
        return {"loss": loss, "image_embed": v, "text_embed": t}


# ---------------------------------------------------------------------------
# Optimizer parameter groups
# ---------------------------------------------------------------------------


def build_param_groups(
    *named_parts: Tuple[str, nn.Module, float],
    weight_decay: float = 0.05,
    no_decay_names: Tuple[str, ...] = (
        "bias",
        "norm",
        "log_t",
        "cls_token",
        "mask_token",
        "embed",
    ),
) -> List[Dict[str, object]]:
    """Build module-specific parameter groups for an optimizer.

    The §2.1 stage-2 recipe assigns "module-specific learning rates and
    decay schedules" to vision, text and projection components. Pass each
    part as a ``(name, module, lr_multiplier)`` triple; this returns a
    list of param-group dicts ready for any ``torch.optim`` constructor.

    Args:
        named_parts: One or more ``(name, module, lr_mult)`` triples. The
            ``lr_mult`` field is stored under ``"lr_mult"`` in each group;
            multiply your base learning rate by it inside the optimizer's
            scheduler step.
        weight_decay: Decay applied to the "decay" group of every part.
        no_decay_names: Substrings that, when present in a parameter
            name, route the parameter into the no-weight-decay group.

    Returns:
        List of parameter-group dicts with keys ``"name"``, ``"params"``,
        ``"weight_decay"`` and ``"lr_mult"``.
    """
    groups = []
    for name, module, lr_mult in named_parts:
        decay = []
        nodecay = []
        for pname, p in module.named_parameters():
            if not p.requires_grad:
                continue
            if any(tag in pname for tag in no_decay_names) or p.ndim == 1:
                nodecay.append(p)
            else:
                decay.append(p)
        if decay:
            groups.append(
                {
                    "name": f"{name}.decay",
                    "params": decay,
                    "weight_decay": weight_decay,
                    "lr_mult": lr_mult,
                }
            )
        if nodecay:
            groups.append(
                {
                    "name": f"{name}.nodecay",
                    "params": nodecay,
                    "weight_decay": 0.0,
                    "lr_mult": lr_mult,
                }
            )
    return groups


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _smoke_test() -> None:
    cfg = CogViTConfig(
        image_size=64,
        patch_size=8,
        embed_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        drop_path_rate=0.1,
    )
    enc = CogViT(cfg)
    x = torch.randn(2, 3, 64, 64)
    out = enc(x)

    print(out["patch_tokens"].shape)
    print(out)


if __name__ == "__main__":
    _smoke_test()
