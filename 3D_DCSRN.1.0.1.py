"""""""""""""""""""""""""""""""""""""""
 Definition of 3D DCSRN
"""""""""""""""""""""""""""""""""""""""

"default conv layer"
def default_conv(in_channels, out_channels, kernel_size, bias = True):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)
    


"Upsampler Module, implemented by employeed of sub-pixel conv"
class Upsampler(nn.Sequential):
    """
    Upsampling/Upscale module, used as last part of "SR reconstruction network model" if the network model employ the "post-upsampling mode".
    Beware the actual upsampling approach is "sub-pixel conv" (which is nn.PixelShuffle() in Pytorch) which was proposed in
    paper: "2016. Real-Time single image and video super-resolution using an efficient sub-pixel convolutional neural network".
    Such sub-pixel conv actually constructs F ∗ S^2 feature maps of dimensions H ×W are reshaped into F feature maps of dimensions H ∗ S × W ∗ S, 
    where S is the upsampling factor.
    """
    def __init__(self, conv, scale, n_feats):
        super(Upsampler, self).__init__()
        if scale == 2:
            self.upsampler = nn.Sequential(*[
                nn.Conv2d(n_feats, n_feats * 4, kernel_size = 3, padding=1, stride=1),
                nn.PixelShuffle(scale)
                ])
        elif scale == 4:
            self.upsampler = nn.Sequential(*[
                nn.Conv2d(n_feats, n_feats * 4, kernel_size = 3, padding=1, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(n_feats, n_feats * 4, kernel_size = 3, padding=1, stride=1),
                nn.PixelShuffle(2),
                ])
        else:
            raise ValueError("scale must be 2 or 4.")

    def forward(self, x):
        upsampled_x = self.upsampler(x)
        return upsampled_x


class BN_Act_Layer(nn.Module):
    def __init__(self, n_feats, bn, act):
        super(BN_Act_Layer, self).__init__()
        bn_act_module=[]
        if bn == True:
            bn_act_module.append(nn.BatchNorm3d(n_feats))
        bn_act_module.append(act)
        
        self.bn_act = nn.Sequential(*bn_act_module)
    
    def forward(self, x):
        return self.bn_act(x)
    
    
class Dense_Block(nn.Module):
    """
    Residual Channel Attention Block (RCAB): There are several RCABs belong to one RG(Residual Group).
    See figure 4 of original RCAN paper
    """
    def __init__(
        self, conv, n_feats, kernel_size, n_layers, bias=True, bn=False, act=nn.ReLU(True)):
        self.n_layers = n_layers
        
        super(Dense_Block, self).__init__()
        self.body = nn.ModuleList()
        self.bn_act = nn.ModuleList()
        for i in range(self.n_layers): 
            self.body.append(conv((2+i)*n_feats, n_feats, kernel_size, bias=bias))
            self.bn_act.append(BN_Act_Layer((2+i)*n_feats, bn, act))


    def forward(self, x):
        for i in range(self.n_layers):
            if i==0:
                y = self.body[i](self.bn_act[i](x))
                res = tc.cat((y,x),dim=1)
            else:
                y = self.body[i](self.bn_act[i](res))
                res = tc.cat((y,res),dim=1)
        return res


class DCSRN_3D(nn.Module):
    def __init__(self, args, not_use_last_conv_to_change_num_channels_to_n_colors = False):
        super(DCSRN_3D, self).__init__()

        conv = default_conv
        act = nn.ELU()        

        self.n_blocks = args['n_blocks'] # number of RGs in RIR/RCAN
        n_layers = args['n_layers'] # number of RCABs in one RG
        n_colors = args['n_colors'] # number of channels going of input of entire model
        n_feats = args['n_feats'] # number of feature maps/channels going through the entire model
        kernel_size = 3 # conv filter size used for all conv in RCAN

        # define commonly use head module
        modules_head = [conv(n_colors, 2*n_feats, kernel_size)] # the first conv layer in RCAN, show in figure 2 of RCAN paper

        self.body = nn.ModuleList()
        for i in range(self.n_blocks):
            self.body.append(Dense_Block(conv, n_feats, kernel_size, n_layers, bn=False, act=act))
        
        self.compressor = nn.ModuleList()
        for i in range(self.n_blocks):
            if i == self.n_blocks-1:
                self.compressor.append(conv((2+6*(i+1))*n_feats,n_colors,kernel_size=1))
            else:
                self.compressor.append(conv((2+6*(i+1))*n_feats, 2*n_feats, kernel_size=1))

        self.head = nn.Sequential(*modules_head)

        self.act = nn.ReLU()

    def forward(self, x):
        y = self.head(x) # input image goes through first conv layer
        for i in range(self.n_blocks):
            if i==0:
                res=tc.cat((y,self.body[i](y)),dim=1)
            else:
                res=tc.cat((res,self.body[i](y)),dim=1)
            
            y = self.compressor[i](res)
            
        return self.act(y)
