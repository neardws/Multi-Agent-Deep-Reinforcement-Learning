import torch

if torch.cuda.device_count() > 1:#判断是不是有多个GPU
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # 就这一行