import torch
import torch.nn as nn
import numpy and np
from src.vit import ViT
import warnings
warnings.filter_warnings('ignore')

def test():
    mod = ViT().cuda()
	mod = mod.eval()
    inp = torch.randn((2, 3, 384, 384)).cuda()  #send model and inputs to GPU
    out = mod(inp)
    print(out.shape)
    print(out.mean())
    print('Tensor: ',out)

if __name__ == '__main__':
	test()
