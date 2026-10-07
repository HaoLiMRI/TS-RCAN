"""
Definition of 2D UNet
"""
"Upsampler Module, implemented by employeed of sub-pixel conv"
class Upsampler(nn.Sequential):
    def __init__(self, in_feats, out_feats, reduce_number_of_channels_in_half = False):
        super(Upsampler, self).__init__()
        self.upsampler = nn.Sequential(*[
                nn.Conv2d(in_feats, 4*in_feats, kernel_size = 3, padding=1, stride=1),
                nn.PixelShuffle(upscale_factor = 2)
                ])
        
        
        if reduce_number_of_channels_in_half == False:
            self.layers = nn.Sequential(*[
                nn.Conv2d(in_feats, in_feats, kernel_size = 3, padding=1, stride=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_feats, out_feats, kernel_size = 3, padding=1, stride=1),
                nn.ReLU(inplace=True)
            ])
        else: # reduce_number_of_channels_in_half == True
            self.layers = nn.Sequential(*[
                nn.Conv2d(2*in_feats, in_feats, kernel_size = 3, padding=1, stride=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_feats, out_feats, kernel_size = 3, padding=1, stride=1),
                nn.ReLU(inplace=True)
            ])

    def forward(self, x, x2=None):
        x = self.upsampler(x)
        if x2 is None:
            x = self.layers(x)
        else:
            x = self.layers(tc.cat((x2, x), dim=1))

        return x


"Downsampler Module, implemented by employeed of sub-pixel conv"
class Downsampler(nn.Sequential):
    def __init__(self, in_feats, out_feats):
        super(Downsampler, self).__init__()
        self.layers = nn.Sequential(*[
                nn.Conv2d(in_feats, out_feats, kernel_size = 3, padding=1, stride=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_feats, out_feats, kernel_size = 3, padding=1, stride=1, bias=False),
                nn.ReLU(inplace=True),
            ])
        self.downsampler = nn.Sequential(*[nn.Conv2d(out_feats, out_feats, kernel_size = 2, padding=0, stride=2, bias=False),
                nn.ReLU(inplace=True),
            ])

    def forward(self, x):
        x2 = self.layers(x)
        out = self.downsampler(x2)
        return out, x2


class UNet_2D(nn.Module):
    def __init__(self, args):
        super(UNet_2D, self).__init__()
        n_colors = args['n_colors']
        n_feats = args['n_feats']
        upscale_factor = args['scale']
        
        self.downsampler_1 = Downsampler(n_colors, n_feats)
        self.downsampler_2 = Downsampler(n_feats, 2*n_feats)
        self.downsampler_3 = Downsampler(2*n_feats, 4*n_feats)
        self.downsampler_4 = Downsampler(4*n_feats, 8*n_feats)
        
        self.mid_layers = nn.Sequential(*[
            nn.Conv2d(8*n_feats, 8*n_feats, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(inplace=True), 
            nn.Conv2d(8*n_feats, 8*n_feats, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(inplace=True)
            ])
        
        self.upsampler_1 = Upsampler(8*n_feats, 4*n_feats, reduce_number_of_channels_in_half=True)
        self.upsampler_2 = Upsampler(4*n_feats, 2*n_feats, reduce_number_of_channels_in_half=True)
        self.upsampler_3 = Upsampler(2*n_feats, n_feats,reduce_number_of_channels_in_half=True)
        self.upsampler_4 = Upsampler(n_feats, n_feats, reduce_number_of_channels_in_half=True)
        
        if Maintain_in_plane_Size == False:
            self.upsampler_5 = Upsampler(n_feats, n_feats, reduce_number_of_channels_in_half=False)
        self.tail = nn.Conv2d(n_feats, n_colors, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x, x1 = self.downsampler_1(x)
        x, x2 = self.downsampler_2(x)
        x, x3 = self.downsampler_3(x)
        x, x4 = self.downsampler_4(x)
        
        x = self.mid_layers(x)
        
        x = self.upsampler_1(x, x4)
        x = self.upsampler_2(x, x3)
        x = self.upsampler_3(x, x2)
        x = self.upsampler_4(x, x1)
        
        if Maintain_in_plane_Size == False:
            x = self.upsampler_5(x)
        
        x = self.tail(x)
        
        return x
    
