# 1.Requirements

```python
conda create --name openmmlab python=3.8 -y 
conda activate openmmlab
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 
pip install timm==0.6.13
pip install mmcv==2.0.0rc4
pip install opencv-python==4.11.0.86
pip install mmsegmentation==1.2.2
```

DDPNet is based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/tree/main) and [CGRSeg](https://github.com/nizhenliang/CGRSeg),The specific method for setting up the environment can be found in the official [mmsegmentation documentation](https://mmsegmentation.readthedocs.io/zh-cn/latest/get_started.html).

# 2.Checkpoint

- The backbone network uses [efficientformerV2](https://github.com/snap-research/EfficientFormer), and the pre-training can be obtained directly from the official
- The weights on **ADE20K** of our proposed DDPNet  are obtained by clicking [here](https://pan.quark.cn/s/145849808e6a)
- The weights on **PASCAL-CONTEXT** of our proposed DDPNet  are obtained by clicking [here](https://pan.quark.cn/s/145849808e6a)

# 3.Traning ＆ Testing

- ## traning

  ```python
  python train.py local_configs/ddpseg/ddpseg_1×b16_160k_ade20k-512×512.py
  ```

- ## Testing 

  ```python
  python test.py local_configs/ddpseg/ddpseg_1×b16_160k_ade20k-512×512.py ${CHECKPOINT_FILE}
  ```

  

