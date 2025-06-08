_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/pascal_context_59.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint = 'checkpoint/mit_b0_20220624-7e0fe6dd.pth'

# 模型配置
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    decode_head=dict(
        type='PromptSegformerHead',
        in_channels=[32, 64, 160, 256],  # MIT-B0的实际输出通道
        channels=256,
        num_classes=59,
        use_point_prompt=True,
        # 点提示相关参数
        prompt_embed_dim=256,
        n_per_side=16,  # 512x512输入下16x16网格
        points_batch_size=32,
        cross_attn_heads=8,
        # 过滤阈值参数
        pred_iou_thresh=0.85,
        stability_score_thresh=0.9,
        box_nms_thresh=0.7))

# 优化器配置
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

# 学习率策略
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=80000,
        by_epoch=False)
]

# 数据加载配置
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True)

test_dataloader = val_dataloader