from functools import partial

import torch
import torch.nn as nn

from mmengine.model.weight_init import kaiming_init, constant_init
import math
from timm.models.layers import trunc_normal_, DropPath
from timm.models.layers.helpers import to_2tuple

class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims """

    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class DDPA(nn.Module):

    def __init__(self, inp, kernel_size=1, ratio=1, band_kernel_size=11, dw_size=(1, 1), padding=(0, 0), stride=1,
                 square_kernel_size=2, relu=True, lambda_init=0.3):
        super(DDPA, self).__init__()

        self.dwconv_hw = nn.Conv2d(inp, inp, square_kernel_size, padding=square_kernel_size // 2, groups=inp)

        self.pool_h1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w1 = nn.AdaptiveAvgPool2d((1, None))

        self.pool_h2 = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w2 = nn.AdaptiveMaxPool2d((1, None))

        gc = inp // ratio

        self.excite1 = nn.Sequential(
            nn.Conv2d(inp, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
            nn.Conv2d(gc, inp, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
            nn.Sigmoid()
        )

        self.excite2 = nn.Sequential(
            nn.Conv2d(inp, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
            nn.Conv2d(gc, inp, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
            nn.Sigmoid()
        )

        self.lambda_h1 = nn.Parameter(torch.zeros(gc, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_w1 = nn.Parameter(torch.zeros(gc, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_h2 = nn.Parameter(torch.zeros(gc, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_w2 = nn.Parameter(torch.zeros(gc, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_init = lambda_init
        self.norm = nn.LayerNorm([inp, 1, 1])

    def sge_diff(self, x):
        # average pooling
        x_h1 = self.pool_h1(x)
        x_w1 = self.pool_w1(x)
        x_gather1 = x_h1 + x_w1
        ge1 = self.excite1(x_gather1)

        # max pooling
        x_h2 = self.pool_h2(x)
        x_w2 = self.pool_w2(x)
        x_gather2 = x_h2 + x_w2
        ge2 = self.excite2(x_gather2)

        lambda_val = torch.exp(torch.sum(self.lambda_h1 * self.lambda_w1)) - \
                     torch.exp(torch.sum(self.lambda_h2 * self.lambda_w2)) + self.lambda_init

        # diff
        diff_ge = ge1 - lambda_val * ge2
        diff_ge = torch.sigmoid(diff_ge)

        return diff_ge

    def forward(self, x):

        loc = self.dwconv_hw(x)

        diff_att = self.sge_diff(x)

        diff_att = self.norm(diff_att.mean(dim=[2, 3], keepdim=True)).expand_as(diff_att)

        out = diff_att * loc

        out = out * (1.0 - self.lambda_init)

        return out


class DDPM(nn.Module):

    def __init__(
            self,
            dim,
            token_mixer=DDPA,
            norm_layer=nn.BatchNorm2d,
            mlp_layer=ConvMlp,
            mlp_ratio=2,
            act_layer=nn.GELU,
            ls_init_value=1e-6,
            drop_path=0.,
            dw_size=11,
            square_kernel_size=3,
            ratio=1,
            lambda_init=0.3,
    ):
        super().__init__()
        self.token_mixer = token_mixer(dim, band_kernel_size=dw_size,
                                   square_kernel_size=square_kernel_size,
                                   ratio=ratio, lambda_init=lambda_init)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x