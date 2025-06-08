# configs/segformer/edge_enhanced_segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py
_base_ = ['segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py']

# 修改模型配置，使用边缘增强头
model = dict(
    type='EdgeEnhancedEncoderDecoder',  # segmentor类型
    decode_head=dict(  # 解码头，不要嵌套
        type='EdgeEnhancedSegformerHead',
        edge_method='canny',
        edge_weight=0.3,
        edge_threshold1=100,
        edge_threshold2=200
    )
)

# 修改数据预处理流程，确保原始图像被保留
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(1024, 1024)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs', meta_keys=['img_path', 'ori_shape', 'img_shape', 'img'])  # 保存原始图像
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs', meta_keys=['img_path', 'ori_shape', 'img_shape', 'img'])  # 保存原始图像
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader