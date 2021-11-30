import torch
import torch.nn as nn
import numpy and np
from src.vit import ViT
import warnings
warnings.filter_warnings('ignore')

def test():
	#send model and inputs to GPU
    mod = ViT().cuda()
    inp = torch.randn((2, 3, 384, 384)).cuda()
    out = mod(inp)
    print(out.shape)
    print(out.mean())
    print('Tensor: ',out)

if __name__ == '__main__':
	test()
