""" File to use Vision Transformer in inference mode """

import torch
import torch.nn as nn
import numpy as np
import argparse
from src.vit import ViT

parser = argparse.ArgumentParser()

parser.add_argument("depth", help = "Integer to determine number of attention blocks")
parser.add_argument("projp", help = "Float value to control dropout probability of output")
parser.add_argument("attnp", help = "Float value to control dropout probability of attention layer")
parser.add_argument("gpu", help = "Boolean to confirm presence of GPU")
parser.add_argument("wts", help = "Path to weights of the trained model")

arg = parser.parse_args()

model = ViT(depth=args.depth, projp=args.projp, attnp=args.attnp)
inp = torch.randn((16, 3, 384, 384))

if args.gpu:    
    model = model.cuda()
    inp = inp.cuda()

model.load_state_dict(torch.load(args.wts))
model.eval()

with torch.no_grad():
  out = model(inp)
 


