# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models import CGRPointSeg
from mmseg.models.decode_heads.dpg_head import DPGHead


def test_dpg_head():
    # 测试参数验证
    with pytest.raises(AssertionError):
        # 验证 pool 参数必须是 'avg' 或 'att'
        DPGHead(in_ch=64, mid_ch=32, pool='invalid', fusions=['channel_add'])

    with pytest.raises(AssertionError):
        # 验证 fusions 参数必须是 'channel_add' 或 'channel_mul'
        DPGHead(in_ch=64, mid_ch=32, pool='avg', fusions=['invalid'])

    with pytest.raises(AssertionError):
        # 验证 fusions 不能为空
        DPGHead(in_ch=64, mid_ch=32, pool='avg', fusions=[])

    # 测试 attention 池化模式
    in_channel = 64
    mid_channel = 32
    batch_size = 2
    height, width = 16, 16
    num_classes = 19

    inputs_x = torch.randn((batch_size, in_channel, height, width))
    inputs_y = torch.randn((batch_size, in_channel, height, width))  # 注意：这里修改为in_channel
    class_features = torch.randn((batch_size, num_classes, height, width))

    # 测试使用 attention 池化
    model_att = DPGHead(
        in_ch=in_channel,
        mid_ch=mid_channel,
        pool='att',
        fusions=['channel_add', 'channel_mul'],
        num_classes=num_classes)

    output_att = model_att(inputs_x, inputs_y)
    # 检查输出 shape
    assert output_att.shape == inputs_x.shape

    # 测试使用平均池化
    model_avg = DPGHead(
        in_ch=in_channel,
        mid_ch=mid_channel,
        pool='avg',
        fusions=['channel_add'],
        num_classes=num_classes)

    output_avg = model_avg(inputs_x, inputs_y)
    # 检查输出 shape
    assert output_avg.shape == inputs_x.shape

    # 测试仅使用 channel_mul 融合
    model_mul = DPGHead(
        in_ch=in_channel,
        mid_ch=mid_channel,
        pool='avg',
        fusions=['channel_mul'],
        num_classes=num_classes)

    output_mul = model_mul(inputs_x, inputs_y)
    # 检查输出 shape
    assert output_mul.shape == inputs_x.shape

    # 测试不同的原型参数
    model_proto = DPGHead(
        in_ch=in_channel,
        mid_ch=mid_channel,
        pool='avg',
        fusions=['channel_add'],
        num_classes=num_classes,
        inter_scale=16,
        top_k=5)

    output_proto = model_proto(inputs_x, inputs_y)
    # 检查输出 shape
    assert output_proto.shape == inputs_x.shape

    # 测试不同分辨率输入
    height2, width2 = 32, 32
    inputs_x2 = torch.randn((batch_size, in_channel, height2, width2))
    inputs_y2 = torch.randn((batch_size, in_channel, height2, width2))  # 注意：这里修改为in_channel

    output2 = model_avg(inputs_x2, inputs_y2)
    # 检查输出 shape
    assert output2.shape == inputs_x2.shape

    # 测试前向传播所有分支
    # 分离测试原型匹配
    enhanced_protos = model_avg.prototype_matching()
    assert enhanced_protos.shape == (num_classes * (model_avg.top_k + 1), in_channel)

    # 分离测试双路径交叉注意力
    updated_protos = model_avg.dual_path_cross_attention(
        inputs_x, class_features, enhanced_protos)  # 这里使用class_features
    assert updated_protos.shape == (batch_size, num_classes * (model_avg.top_k + 1), in_channel)

    # 分离测试生成提示
    prompts = model_avg.generate_prompts(inputs_x, updated_protos)
    assert prompts.shape == inputs_x.shape

    # 测试GPU运行
    if torch.cuda.is_available():
        model_cuda = DPGHead(
            in_ch=in_channel,
            mid_ch=mid_channel,
            pool='avg',
            fusions=['channel_add'],
            num_classes=num_classes).cuda()

        inputs_x_cuda = inputs_x.cuda()
        inputs_y_cuda = inputs_y.cuda()

        output_cuda = model_cuda(inputs_x_cuda, inputs_y_cuda)
        assert output_cuda.shape == inputs_x_cuda.shape


def test_cgr_point_seg():
    # 基本参数配置
    in_channels = [32, 64, 128]
    channels = 64
    num_classes = 19

    # 使用BN而非SyncBN
    norm_cfg = dict(type='BN', requires_grad=True)

    # 创建模型
    model = CGRPointSeg(
        in_channels=in_channels,
        channels=channels,
        num_classes=num_classes,
        in_index=[0, 1, 2],
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,  # 使用BN而非默认的SyncBN
        is_dw=True,
        dw_size=7,
        neck_size=3,
        next_repeat=4,
        square_kernel_size=1,
        module='RCA',
        ratio=1,
        dpg_pool='att',
        dpg_fusions=['channel_mul', 'channel_add'],
        dpg_top_k=3,
        dpg_inter_scale=8
    )

    # 检查是否还有SyncBN
    for module in model.modules():
        assert not isinstance(module, torch.nn.SyncBatchNorm), "模型中仍有SyncBatchNorm"

    # 创建测试输入
    inputs = [
        torch.randn(2, 32, 32, 32),
        torch.randn(2, 64, 16, 16),
        torch.randn(2, 128, 8, 8)
    ]

    # 执行前向传播
    output = model(inputs)

    # 验证输出形状
    expected_shape = (2, num_classes, 32, 32)
    assert output.shape == expected_shape

    # 如果有GPU可用，测试GPU模式
    if torch.cuda.is_available():
        model = model.cuda()
        inputs_cuda = [inp.cuda() for inp in inputs]
        output_cuda = model(inputs_cuda)
        assert output_cuda.shape == expected_shape