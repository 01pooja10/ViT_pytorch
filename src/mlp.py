class MLP(nn.module):
    '''Multilayer perceptron'''

    def __init__(self, feat, hidden, p=0):
        super(MLP,self).__init__()

        self.hl = hidden
        self.feat = feat
        self.p = p

        self.l1 = nn.Linear(self.feat, hidden)
        self.l2 = nn.Linear(hidden, self.feat)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(p)

    def forward(self, x):
        '''x.shape - batch, npatches+1, input features'''

        x = self.l1(x)
        x = self.gelu(x)
        x = self.drop(x)

        x = self.l2(x)
        x = self.drop(x)
		
		return x
