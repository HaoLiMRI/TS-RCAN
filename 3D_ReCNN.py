"""
Definition of 3D ReCNN
"""
    
class RECNN(nn.Module):
    def __init__(self, args):
        super(RECNN, self).__init__()
        
        n_colors = args['n_colors']
        n_layers = args['n_layers']-2
        n_feats = args['n_feats']
        
        self.head = nn.Sequential(*[nn.Conv3d(in_channels=n_colors, out_channels=n_feats, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True)])
        self.tail = nn.Sequential(*[nn.Conv3d(in_channels=n_feats, out_channels=n_colors, kernel_size=3, stride=1, padding=1)])
        body = []
        for i in range(n_layers):
            body.append(nn.Conv3d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=1, padding=1))
            body.append(nn.ReLU(inplace=True))
        self.body = nn.Sequential(*body)


    def forward(self, x):
        res = self.head(x)
        res = self.body(res)
        res = self.tail(res)
        return x+res
    
