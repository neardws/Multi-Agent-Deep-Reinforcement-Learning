from tqdm import tqdm
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

@torch.jit.script
def foo():
    x = torch.ones((1024 * 12, 1024 * 12), dtype=torch.float32).cuda()
    y = torch.ones((1024 * 12, 1024 * 12), dtype=torch.float32).cuda()
    z = x + y
    return z


if __name__ == '__main__':
    z0 = None
    for _ in tqdm(range(10000000000)):
        zz = foo()
        if z0 is None:
            z0 = zz
        else:
            z0 += zz