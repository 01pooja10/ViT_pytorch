""" File to use Vision Transformer in inference mode """

import torch
import torch.nn as nn
import numpy as np
import argparse
from src import vit

parser = argparse.ArgumentParser()
parser.add_argument("Output Dropout Probability", help = "Argument to control dropout probability of output")
parser.add_argument("Attention Dropout Probability", help = "Argument to control dropout probability of attention layer")
arg = parser.parse_args()
