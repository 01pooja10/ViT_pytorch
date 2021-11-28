class ViT(nn.module):
    def __init__(self, img_size=384,
                       patch_size=16,
                       channels=3,
                       nclasses=100,
                       emb_dim=768,
                       depth=12,
                       nheads=12,
                       mlp=4,
                       kqv=True,
                       projp=0,
                       attnp=0
                ):

        super(vit,self).__init__()
        self.pe = Embed(img_size, patch_size, channels, emb_dim)
        self.ctoken = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos = nn.Parameter(torch.zeros(1, 1+self.pe.npatches, emb_dim))
        self.pdrop = nn.Dropout(projp)
		
        self.blocks = nn.ModuleList([
                                    Blocks(emb_dim, nheads,
                                    mlp, kqv, projp, attnp)

                                    for _ in range(depth)
                                    ])

        self.norm = nn.LayerNorm(emb_dim, eps=1e-6)
        self.flin = nn.Linear(emb_dim, nclasses)

    def forward(self, x):
        """ x: (batch_size, channels, img_size, img_size) """

        bs = x.shape[0]
        x = self.pe(x)
        ctoken = self.ctoken.expand(bs, -1, -1)  #(batch_size, 1, emb_dim)
        x = torch.cat((ctoken, x),dim=1)   #(batch_size, 1+npatches, emb_dim)

        x = x + self.pos
        x = self.pdrop(x)

        for b in self.blocks:
            x = b(x)

        x = self.norm(x)
        cl = x[:, 0]        #select only class embedding
        cl = self.flin(cl)

        return cl
