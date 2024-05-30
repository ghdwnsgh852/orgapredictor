# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer

import torch.nn.functional as F


import timm

assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_

from pos_embed import interpolate_pos_embed








class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model




class CustomModel(nn.Module):

    def __init__(self, ckpt=None, global_pool=False):
        super().__init__()
        model = vit_base_patch16(num_classes=2,
                                 drop_path_rate=0.1,
                                 global_pool=False,
                                 img_size=512)
        if ckpt:

            self.model = self._load_weights(model, ckpt)
        else:
            self.model = model
        self.fc = nn.Linear(768 * 2, 2)

    def _load_weights(self, model, ckpt):
        checkpoint = torch.load(ckpt)

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()

        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[
                    k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        model.load_state_dict(checkpoint_model, strict=False)

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)
        return model

    def forward(self, img1, img2):

        x1 = self.model.forward_features(img1)
        x2 = self.model.forward_features(img2)
        x = torch.cat([x1, x2], dim=1)
        x = self.fc(x)

        return x

    def attention_map(self, img1, img2):

        x1 = self.model.forward_features(img1)
        x2 = self.model.forward_features(img2)

        x = torch.cat([x1, x2], dim=1)

        return x



class CustomMinusModel(nn.Module):
    def __init__(self, ckpt=None, global_pool=False):
        super().__init__()
        model=vit_base_patch16(
        num_classes=2,
        drop_path_rate=0.1,
        global_pool=False,
        img_size=512
        )
        self.model=self._load_weights(model,ckpt)
        self.fc=nn.Linear(768,2)



    def _load_weights(self, model, ckpt):
        checkpoint=torch.load(ckpt)

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()

        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        interpolate_pos_embed(model, checkpoint_model)
        model.load_state_dict(checkpoint_model, strict=False)
        trunc_normal_(model.head.weight, std=2e-5)
        return model


    def forward(self, img1, img2):

        x1=self.model.forward_features(img1)
        x2=self.model.forward_features(img2)
        x=torch.abs(x1-x2)
        x=self.fc(x)

        return x





    def attention_map(self, img1, img2):

        x1=self.model.forward_features(img1)
        x2=self.model.forward_features(img2)
        x=torch.abs(x1-x2)

        return x
