# model settings
# norm_cfg = dict(type='SyncBN', requires_grad=True)
norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='MixVisionTransformer',  # SegFormer的混合视觉Transformer
        in_channels=3,  # 输入通道数（RGB）
        embed_dims=32,  # 初始嵌入维度
        num_stages=4,  # 编码器阶段数
        num_layers=[2, 2, 2, 2],  # 各阶段的Transformer块数量
        num_heads=[1, 2, 5, 8],  # 各阶段注意力头数
        patch_sizes=[7, 3, 3, 3],  # 各阶段补丁大小
        sr_ratios=[8, 4, 2, 1],  # 各阶段空间缩减比率（K/V下采样率）
        out_indices=(0, 1, 2, 3),  # 输出多级特征图索引
        mlp_ratio=4,  # MLP扩展比率
        qkv_bias=True,  # 在qkv计算中添加可学习偏置
        drop_rate=0.0,  # 随机丢弃率
        attn_drop_rate=0.0,  # 注意力丢弃率
        drop_path_rate=0.1),  # 随机深度衰减率
    decode_head=dict(
        type='SegformerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,  # 双线性上采样对齐方式
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    # 测试模式（whole:全图推理，slide:滑动窗口)
    test_cfg=dict(mode='whole'))
