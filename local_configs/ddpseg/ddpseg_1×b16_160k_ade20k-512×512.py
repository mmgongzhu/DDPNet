_base_ = [
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py',
]

crop_size = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    size=crop_size,
    pad_val=0,
    seg_pad_val=255
)

# norm_cfg = dict(type='SyncBN', requires_grad=True)
norm_cfg = dict(type='BN', requires_grad=True)


checkpoint = 'checkpoint/eformer_s1_450.pth'

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='efficientformerv2_s1_feat',
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
    ),
    decode_head=dict(
        type='DDPSeg',
        in_channels=[48, 120, 224],
        in_index=[1, 2, 3],
        channels=128,
        dropout_ratio=0.1,
        num_classes=150,
        is_dw=True,
        dw_size=11,
        neck_size=11,
        next_repeat=5,
        square_kernel_size=3,
        ratio=1,
        module='DDPA',
        use_diff=True,
        lambda_init=None,
        norm_cfg=norm_cfg,
        align_corners=False,
         loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00012, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }
    )
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

train_dataloader = dict(batch_size=16, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
find_unused_parameters = True
randomness = dict(seed=0, deterministic=False)