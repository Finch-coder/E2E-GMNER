"""Runtime utilities shared by training and evaluation."""

import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch

def seed_everything(seed: Optional[int] = 42, *, verbose: bool = True):

    #随机种子固定
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)  #torch CPU随机
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed) #GPU随机
    
    #deterministic_algorithms、 torch.backends.cudnn.benchmark如果设置为强制确定性 会导致训练异常缓慢
    #尽管这些加速可能会 多次训练的曲线/指标也可能有小幅波动，但对最终指标通常影响不大
    # 🚀 性能优先（关闭确定性）
    torch.backends.cudnn.benchmark = True #cuDNN 为当前输入形状自动搜索最快的卷积/算子实现。 若
    torch.backends.cudnn.deterministic = False  

    # 🚀 允许 TF32（Ampere+ 巨幅加速）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ❌ 不再使用 deterministic algorithms
    torch.use_deterministic_algorithms(False) #避免因“必须确定性”导致的慢速

def get_first_param_device(model) -> torch.device:
    return next(model.parameters()).device


def supports_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def pick_vision_inputs(batch: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k in ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]:
        if k in batch:
            out[k] = batch[k]
    return out

def append_text_log(path: str, line: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")
