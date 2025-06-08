#!/usr/bin/env python
# 专为 CGRPointSeg 模型定制的参数量和计算量计算脚本

import argparse
import os
import tempfile
from pathlib import Path
import sys

import torch
import torch.nn as nn
from mmengine import Config
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope

from mmseg.registry import MODELS
from mmseg.structures import SegDataSample

# 导入计算复杂度的函数
from mmengine.analysis import get_model_complexity_info
from mmengine.analysis.print_helper import _format_size

# 重定向打印函数，屏蔽模型中可能的打印语句
original_print = print


def silent_print(*args, **kwargs):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description='计算 CGRPointSeg 模型的参数量和计算量')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--shape', type=int, nargs='+', default=[480, 480],
                        help='输入图像大小，默认为 480x480（Pascal Context 数据集裁剪大小）')
    parser.add_argument('--silent', action='store_true', help='静默模式，屏蔽模型中的打印语句')
    parser.add_argument('--verbose', action='store_true', help='详细模式，显示模型结构')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger = MMLogger.get_instance(name='CGRPointSeg计算器')

    # 配置文件路径
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f'配置文件 {config_path} 不存在')
        return

    # 加载配置
    cfg = Config.fromfile(config_path)
    cfg.work_dir = tempfile.TemporaryDirectory().name
    cfg.log_level = 'WARN'

    # 初始化默认作用域
    init_default_scope(cfg.get('scope', 'mmseg'))

    # 设置输入形状
    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError('输入形状无效，应为一个或两个整数')

    # 构建模型
    model = MODELS.build(cfg.model)
    if hasattr(model, 'auxiliary_head'):
        model.auxiliary_head = None

    # 添加 hook 来捕获中间输出的张量，帮助调试
    if args.verbose:
        tensor_shapes = []

        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                tensor_shapes.append(f"{module.__class__.__name__}: {output.shape}")
            elif isinstance(output, (list, tuple)) and all(isinstance(o, torch.Tensor) for o in output):
                tensor_shapes.append(f"{module.__class__.__name__}: {[o.shape for o in output]}")

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.register_forward_hook(hook_fn)

    # 屏蔽模型中可能的打印语句
    if args.silent:
        sys.stdout = open(os.devnull, 'w')

    # 移至 GPU（如果可用）
    if torch.cuda.is_available():
        model.cuda()

    # 将 SyncBN 转换为 BN
    model = revert_sync_batchnorm(model)

    # 准备输入数据
    result = {'ori_shape': input_shape[-2:], 'pad_shape': input_shape[-2:]}
    data_batch = {
        'inputs': [torch.rand(input_shape)],
        'data_samples': [SegDataSample(metainfo=result)]
    }
    data = model.data_preprocessor(data_batch)

    # 设为评估模式
    model.eval()

    # 计算复杂度
    try:
        # 临时禁用打印
        if args.silent:
            globals()['print'] = silent_print

        outputs = get_model_complexity_info(
            model,
            input_shape=None,
            inputs=data['inputs'],
            show_table=args.verbose,
            show_arch=args.verbose
        )

        # 恢复打印功能
        if args.silent:
            globals()['print'] = original_print
            sys.stdout = sys.__stdout__

        # 打印结果
        split_line = '=' * 30
        print(f'{split_line}')
        print(f'模型: {cfg.model.decode_head.type}')
        print(f'输入形状: {input_shape[-2:]}')
        print(f'计算量: {_format_size(outputs["flops"])}')
        print(f'参数量: {_format_size(outputs["params"])}')
        print(f'{split_line}')

        if args.verbose and tensor_shapes:
            print("中间张量形状:")
            for shape in tensor_shapes:
                print(f"  {shape}")
            print(f'{split_line}')

    except Exception as e:
        if args.silent:
            sys.stdout = sys.__stdout__
        print(f"计算复杂度时出错: {e}")


if __name__ == '__main__':
    main()