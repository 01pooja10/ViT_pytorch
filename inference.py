""" File to use Vision Transformer in inference mode """

import torch
import torch.nn as nn
import numpy as np
import argparse
from src.vit import ViT

parser = argparse.ArgumentParser()
parser.add_argument("Image Size", help = "Argument to determine image size")
parser.add_argument("Patch Size", help = "Argument to determine patch size")
parser.add_argument("Transformer Depth", help = "Argument to determine number of attention blocks")
parser.add_argument("Output Dropout Probability", help = "Argument to control dropout probability of output")
parser.add_argument("Attention Dropout Probability", help = "Argument to control dropout probability of attention layer")
arg = parser.parse_args()

