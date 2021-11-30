import torch
import torch.nn as nn
from src.mlp import MLP
from src.attn import Attention


class Blocks(nn.module):
    '''Reusable transformer blocks'''

    def __init__(self, dim, nheads, mlpf, kqv=True, projp, attnp):
        super(Blocks,self).__init__()

        self.dim = dim
        self.nheads = nheads
        self.mlp = mlpf
        self.kqvbias = kqv
        self.projp = projp
        self. attnp = attnp

        self.lnorm1 = nn.LayerNorm(self.dim, eps=1e-6)
        self.lnorm2 = nn.LayerNorm(self.dim, eps=1e-6)

        self.attn = Attention(self.dim, self.nheads,
                            self.kqv, self.projp, self.attnp)
        self.hid = int(self.dim * self.mlpf)
        self.mlp = MLP(self.dim, self.hid)

    def forward(self, x):
        '''x.shape - (batch, npatches+1, dim)'''

        x = x + self.attn(self.lnorm1(x))
        x = x + self.mlp(self.lnorm2(x))
        return x
