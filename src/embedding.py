import torch
import torch.nn as nn

class Embed(nn.module):

    def __init__(self, img_size, patch_size, channels=3, emb_dim=768):
        super(Embed,self).__init__()
        self.img_s = img_size
        self.patch_s = patch_size
        self.chn = channels
        self.emb = emb_dim
        self.npatches = (img_size//patch_size) ** 2
        self.project = nn.Conv2d(channels, emb_dim,
                                kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.project(x)
        x = x.flatten(2)
        x = x.transpose(1,2)
        return x
