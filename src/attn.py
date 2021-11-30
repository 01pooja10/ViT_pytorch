import torch
import torch.nn as nn

class Attention(nn.module):

    def __init__(self, dim, nheads=12, kqv=True, projp, attnp):
        super(Attention, self).__init__()
        self.nheads = nheads
        self.dim = dim
        self.kqvbias = kqv
        self.projp = projp
        self.atnp = attnp

        self.headdim = dim//nheads
        self.scaler = self.headdim **-0.5
		
        self.lin = nn.Linear(dim,dim*3,bias=kqv)
        self.dropa = nn.Dropout(p=attnp)
        self.proj = nn.Linear(dim,dim)
        self.dropp = nn.Dropout(p=projp)
        self.sm = nn.Softmax(dim=-1)

    def forward(self, x):
        bs, tokens, dim = x.shape
        #kqv for calculating weights
        kqv = self.lin(x)
		
        kqv = kqv.reshape(bs, tokens, 3, self.nheads, self.headdim)
        kqv = kqv.permute(2,0,3,1,4)
        q, k, v = kqv[0], kqv[1], kqv[2]
        dp = (q @ k.transpose(-2, -1)) * self.scaler
		
        attn = self.sm(dp)
        attn = self.dropa(attn)
        wt = attn @ v
        wt = wt.transpose(2,1)     #(batch, patches+1, heads, head_dim)
        #flatten out last 2 dims
        wt = wt.flatten(2)         #(batch, patches+1, dim)

        x = self.proj(wt)
        x = self.dropp(x)

        return x
