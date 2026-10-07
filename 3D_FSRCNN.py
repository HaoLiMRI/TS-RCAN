"""
Definition of 3D FSRCNN
"""
class FSRCNN_3D(nn.Module):
    def __init__(self, args):
        super(FSRCNN_3D, self).__init__()
        
        num_channels = args['n_colors']
        d = args['n_feats']
        s = 12
        m = 4

        self.first_part = nn.Sequential(nn.Conv3d(in_channels=num_channels, out_channels=d, kernel_size=5, stride=1, padding=2),
                                        nn.PReLU())

        self.layers = []
        self.layers.append(nn.Sequential(nn.Conv3d(in_channels=d, out_channels=s, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))
        for _ in range(m):
            self.layers.append(nn.Conv3d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1))
            self.layers.append(nn.PReLU())
        self.layers.append(nn.Sequential(nn.Conv3d(in_channels=s, out_channels=d, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))

        self.mid_part = nn.Sequential(*self.layers)

        self.last_part = nn.Conv3d(in_channels=d, out_channels=num_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.first_part(x)
        out = self.mid_part(out)
        out = self.last_part(out)
        return out
    
