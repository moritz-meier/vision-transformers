from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as functional

import einops
import einops.layers.torch as einops_layers


class VisionTransformer(nn.Sequential):
    """
    * Input Shape: `(batch_size, n_channels, h, w)`
    * Output Shape: `(batch_size, n_classes)`
    """

    def __init__(
            self,
            img_size,
            patch_size,
            *,
            n_layers,
            n_heads,
            n_classes,
            token_size=None,
            mlp_expansion=4,
            p_dropout=0.1,
            **kwargs):

        img_size = img_size[-3:] if type(img_size) is tuple else (img_size, img_size)
        img_size = img_size if len(img_size) == 3 else (1, *(img_size)) if len(img_size) == 2 else (1, *(2 * img_size))

        patch_size = patch_size[-2:] if type(patch_size) is tuple else (patch_size, patch_size)
        patch_size = patch_size if len(patch_size) == 2 else (*(2 * patch_size), )

        if (img_size[1] % patch_size[0] != 0) or (img_size[2] % patch_size[1] != 0):
            raise ValueError("image_size is not a multiple of the patch_size!")

        self.img_size = img_size
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_classes = n_classes
        self.token_size = token_size or patch_size[0] * patch_size[1]
        self.mlp_expansion = mlp_expansion
        self.p_dropout = p_dropout

        self.n_patches = (img_size[1] // patch_size[0]) * (img_size[2] // patch_size[1])
        self.n_tokens = self.n_patches + 1

        super().__init__(
            *[
                PatchEmbedding(self.img_size, self.patch_size, token_size=self.token_size),
                Encoder(
                    n_layers=self.n_layers,
                    n_heads=self.n_heads,
                    n_tokens=self.n_tokens,
                    token_size=self.token_size,
                    mlp_expansion=self.mlp_expansion,
                    p_dropout=self.p_dropout),
                Classifier(n_classes=self.n_classes, token_size=self.token_size)
            ])


class PatchEmbedding(nn.Module):

    def __init__(self, img_size, patch_size, *, token_size=None):
        super().__init__()

        img_size = img_size[-3:] if type(img_size) is tuple else (img_size, img_size)
        img_size = img_size if len(img_size) == 3 else (1, *(img_size)) if len(img_size) == 2 else (1, *(2 * img_size))

        patch_size = patch_size[-2:] if type(patch_size) is tuple else (patch_size, patch_size)
        patch_size = patch_size if len(patch_size) == 2 else (*(2 * patch_size), )

        if (img_size[1] % patch_size[0] != 0) or (img_size[2] % patch_size[1] != 0):
            raise ValueError("image_size is not a multiple of the patch_size!")

        self.img_size = img_size
        self.patch_size = patch_size
        self.token_size = token_size or self.patch_size[0] * self.patch_size[1]

        self.n_patches = (img_size[1] // patch_size[0]) * (img_size[2] // patch_size[1])
        self.n_tokens = self.n_patches + 1

        self.__patch_emb = nn.Sequential(
            *[
                nn.Conv2d(self.img_size[0], self.token_size, kernel_size=self.patch_size, stride=self.patch_size),
                einops_layers.Rearrange(
                    "b token_size n_v_patches n_h_patches -> b (n_v_patches n_h_patches) token_size")
            ])

        # Alternative: Rearrange into patches and apply linear layer to each patch, sum over all input channels
        """
        self.patch_emb = nn.Sequential(
            *[
                ein_layers.Rearrange(
                    "b c (n_v_patches v_patch_size) (n_h_patches h_patch_size) -> b c (n_v_patches n_h_patches) (v_patch_size h_patch_size)",
                    v_patch_size=self.patch_size,
                    h_patch_size=self.patch_size),
                nn.Linear(self.patch_size**2, self.token_size),
                ein_layers.Reduce("b c n_patches token_size -> b n_patches token_size", reduction="sum")
            ])
        """

        self.cls_token = nn.parameter.Parameter(torch.randn((1, 1, self.token_size)))
        self.positions = nn.parameter.Parameter(torch.randn((self.n_tokens, self.token_size)))

    def forward(self, x):
        """
        * Input Shape: `(batch_size, channels, height, width)`
        * Output Shape: `(batch_size, n_tokens, token_size)`
        """

        x = self.__patch_emb(x)
        cls_tokens = einops.repeat(
            self.cls_token, "() n_tokens token_size -> batch_size n_tokens token_size", batch_size=x.shape[0])
        x = torch.concat((cls_tokens, x), dim=1)
        x += self.positions
        return x


class Encoder(nn.Sequential):
    """
    * Input Shape: `(batch_size, n_tokens, token_size)`
    * Output Shape: `(batch_size, n_tokens, token_size)`
    """

    def __init__(self, *, n_layers, **kwargs):

        self.n_layers = n_layers

        super().__init__(*[AttentionBlock(**kwargs) for _ in range(self.n_layers)])


class Classifier(nn.Module):

    def __init__(self, *, n_classes, token_size):
        super().__init__()

        self.n_classes = n_classes
        self.token_size = token_size

        self.__classifier = nn.Sequential(nn.Linear(self.token_size, self.n_classes), nn.LogSoftmax(dim=-1))

    def forward(self, x):
        """
        * Input Shape: `(batch_size, n_tokens, token_size)`
        * Output Shape: `(batch_size, n_classes)`
        """

        return self.__classifier(x[:, 0])


class AttentionBlock(nn.Sequential):
    """
    * Input Shape: `(batch_size, n_tokens, token_size)`
    * Output Shape: `(batch_size, n_tokens, token_size)`
    """

    def __init__(self, *, n_heads, n_tokens, token_size, mlp_expansion=4, p_dropout=0.1):

        self.n_heads = n_heads
        self.n_tokens = n_tokens
        self.token_size = token_size
        self.mlp_expansion = mlp_expansion
        self.p_dropout = p_dropout

        attention_block = Residual(
            nn.Sequential(
                *[
                    nn.LayerNorm(self.token_size),
                    MultiHeadAttention(n_heads=self.n_heads, n_tokens=self.n_tokens, token_size=self.token_size),
                    nn.Dropout(self.p_dropout)
                ]))

        mlp_block = Residual(
            nn.Sequential(
                *[
                    nn.LayerNorm(self.token_size),
                    nn.Linear(self.token_size, self.token_size * self.mlp_expansion),
                    nn.GELU(),
                    nn.Dropout(self.p_dropout),
                    nn.Linear(self.token_size * self.mlp_expansion, self.token_size),
                    nn.Dropout(self.p_dropout)
                ]))

        super().__init__(attention_block, mlp_block)


class MultiHeadAttention(nn.Module):

    def __init__(self, *, n_heads, n_tokens, token_size):
        super().__init__()

        self.n_heads = n_heads
        self.n_tokens = n_tokens
        self.token_size = token_size

        self.__heads = nn.ModuleList(
            [AttentionHead(n_tokens=self.n_tokens, token_size=self.token_size) for i in range(n_heads)])

        self.W = nn.Linear(self.n_heads * self.token_size, self.token_size)

    def forward(self, x):
        """
        * Input Shape: `(batch_size, n_tokens, token_size)`
        * Output Shape: `(batch_size, n_tokens, token_size)`
        """

        x = einops.rearrange(
            [head(x) for head in self.__heads], "n_heads b n_patches token_size -> b n_patches (n_heads token_size)")
        x = self.W(x)
        return x


class AttentionHead(nn.Module):

    def __init__(self, *, n_tokens, token_size):
        super().__init__()

        self.n_tokens = n_tokens
        self.token_size = token_size

        self.scaler = self.token_size**0.5

        self.__queries = nn.Linear(self.token_size, self.token_size)
        self.__keys = nn.Linear(self.token_size, self.token_size)
        self.__values = nn.Linear(self.token_size, self.token_size)

    def forward(self, x):
        """
        * Input Shape: `(batch_size, n_tokens, token_size)`
        * Output Shape: `(batch_size, n_tokens, token_size)`
        """

        q, k, v = self.__queries(x), self.__keys(x), self.__values(x)
        k_t = einops.rearrange(k, "b n_tokens token_size -> b token_size n_tokens")
        attn_matrix = functional.softmax((q @ k_t) / self.scaler, dim=-1)
        return attn_matrix @ v


class Residual(nn.Module):

    def __init__(self, layer):
        super().__init__()

        self.__layer = layer

    def forward(self, x):
        y = self.__layer(x)
        return x + y
