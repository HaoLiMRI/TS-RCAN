"""
Definition of 3D SRCNN
"""
class SRCNN_3D(nn.Module):
    def __init__(self, args):
        super(SRCNN_3D, self).__init__()
        num_channels = args['n_colors']
        base_filter = args['n_feats']
        
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels=num_channels, out_channels=base_filter, kernel_size=9, stride=1, padding=4, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=base_filter, out_channels=base_filter // 2, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=base_filter // 2, out_channels=num_channels, kernel_size=5, stride=1, padding=2, bias=True),
        )

    def forward(self, x):
        out = self.layers(x)
        return out
    
