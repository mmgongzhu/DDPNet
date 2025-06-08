from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from mmseg.models.utils import resize
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .dpg_head import DPGHead
from .ddpm import DDPM, DDPA


def lambda_init_fn(depth_idx):
    return 0.7 - 0.5 * math.exp(-0.3 * depth_idx)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class PyramidPoolAgg(nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride

    def forward(self, inputs):
        B, C, H, W = inputs[-1].shape
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        return torch.cat([nn.functional.adaptive_avg_pool2d(inp, (H, W)) for inp in inputs], dim=1)


class FuseBlockMulti(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            stride: int = 1,
            norm_cfg=dict(type='BN', requires_grad=True),
            activations=None,
    ) -> None:
        super(FuseBlockMulti, self).__init__()
        self.stride = stride
        self.norm_cfg = norm_cfg
        assert stride in [1, 2]

        if activations is None:
            activations = nn.ReLU

        self.fuse1 = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.fuse2 = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.act = h_sigmoid()

    def forward(self, x_l, x_h):
        B, C, H, W = x_l.shape
        inp = self.fuse1(x_l)
        sig_act = self.fuse2(x_h)
        sig_act = F.interpolate(self.act(sig_act), size=(H, W), mode='bilinear', align_corners=False)
        out = inp * sig_act
        return out


class NextLayer(nn.Module):
    def __init__(self, block_num, embedding_dim, dw_size, module=DDPM, mlp_ratio=2, token_mixer=DDPA, square_kernel_size=3, ratio=1):
        super().__init__()
        self.block_num = block_num

        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            def lambda_init_fn(depth):
                return 0.7 - 0.5 * math.exp(-0.3 * depth)

            if isinstance(module, type) and issubclass(module, DDPM) or module == 'DDPM':
                adaptive_lambda = lambda_init_fn(i)
                self.transformer_blocks.append(module(embedding_dim,
                                                      dw_size=dw_size,
                                                      mlp_ratio=mlp_ratio,
                                                      square_kernel_size=square_kernel_size,
                                                      lambda_init=adaptive_lambda,
                                                      ratio=ratio))
            else:
                self.transformer_blocks.append(module(embedding_dim,
                                                      token_mixer=token_mixer,
                                                      dw_size=dw_size,
                                                      mlp_ratio=mlp_ratio,
                                                      square_kernel_size=square_kernel_size,
                                                      ratio=ratio))
    def forward(self, x):
        # token * N
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x


class SpatialGatherModule(nn.Module):
    """Aggregate the context features according to the initial predicted
    probability distribution.

    Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, scale):
        super(SpatialGatherModule, self).__init__()
        self.scale = scale

    def forward(self, feats, probs):
        """Forward function."""
        batch_size, num_classes, height, width = probs.size()
        channels = feats.size(1)
        probs = probs.view(batch_size, num_classes, -1)
        feats = feats.view(batch_size, channels, -1)
        feats = feats.permute(0, 2, 1)
        probs = F.softmax(self.scale * probs, dim=2)
        ocr_context = torch.matmul(probs, feats)
        ocr_context = ocr_context.permute(0, 2, 1).contiguous().unsqueeze(3)
        return ocr_context


@MODELS.register_module()
class DDPSeg(BaseDecodeHead):
    def __init__(self, is_dw=False, next_repeat=4, mr=2, dw_size=7, neck_size=3, square_kernel_size=1, module='DDPA',
                 ratio=1, use_diff=True, lambda_init=None, **kwargs):
        super(DDPSeg, self).__init__(input_transform='multiple_select', **kwargs)
        embedding_dim = self.channels

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=1,
            stride=1,
            groups=embedding_dim if is_dw else 1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        self.ppa = PyramidPoolAgg(stride=2)
        norm_cfg = dict(type='SyncBN', requires_grad=True)
        act_layer = nn.ReLU6

        module_dict = {
            'DDPA': DDPA,
        }
        ddpm_module = DDPM
        token_mixer_type = 'DDPA'

        self.trans = NextLayer(next_repeat, sum(self.in_channels), dw_size=neck_size,
                               module=ddpm_module, mlp_ratio=mr,
                               square_kernel_size=square_kernel_size, ratio=ratio)

        self.context_align = ConvModule(
            in_channels=sum(self.in_channels),
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

        self.ddpm_blocks = nn.ModuleList()
        for i in range(1, len(self.in_channels)):
            adaptive_lambda = lambda_init_fn(i) if lambda_init is None else lambda_init
            if use_diff:
                self.ddpm_blocks.append(
                    DDPM(self.in_channels[i], dw_size=dw_size, mlp_ratio=mr,
                            square_kernel_size=square_kernel_size, ratio=ratio,
                            lambda_init=adaptive_lambda)
                )
            else:
                self.ddpm_blocks.append(
                    DDPM(self.in_channels[i], token_mixer=module_dict[module],
                        dw_size=dw_size, mlp_ratio=mr, square_kernel_size=square_kernel_size,
                        ratio=ratio)
                )

        self.spatial_convs = nn.ModuleList()
        for i in range(3):
            self.spatial_convs.append(
                ConvModule(
                    in_channels=self.channels if i > 0 else self.in_channels[0],
                    out_channels=self.channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )

        self.align_convs = nn.ModuleList()
        for i in range(1, len(self.in_channels)):
            self.align_convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=None
                )
            )

        self.spatial_gather_module = SpatialGatherModule(1)
        self.lgc = DPGHead(embedding_dim, embedding_dim, pool='att', fusions=['channel_mul'])

    def forward(self, inputs):
        xx = self._transform_inputs(inputs)

        context_features = self.ppa(xx)
        context_features = self.trans(context_features)

        aligned_context = self.context_align(context_features)

        x1 = self.spatial_convs[0](xx[0])

        processed_block2 = self.ddpm_blocks[0](xx[1])
        aligned_block2 = self.align_convs[0](processed_block2)
        upsampled_block2 = resize(aligned_block2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x1 = x1 + upsampled_block2

        x2 = self.spatial_convs[1](x1)

        processed_block3 = self.ddpm_blocks[1](xx[2])
        aligned_block3 = self.align_convs[1](processed_block3)
        upsampled_block3 = resize(aligned_block3, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x2 = x2 + upsampled_block3

        x3 = self.spatial_convs[2](x2)

        upsampled_context = resize(aligned_context, size=x3.shape[2:], mode='bilinear', align_corners=False)
        final_feature = x3 + upsampled_context

        prev_output = self.cls_seg(final_feature)
        context = self.spatial_gather_module(final_feature, prev_output)
        object_context = self.lgc(final_feature, context) + final_feature
        output = self.cls_seg(object_context)

        return output