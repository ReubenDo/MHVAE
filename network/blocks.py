#   CODE ADAPTED FROM: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/network_architecture/generic_UNet.py

from copy import deepcopy
from torch import nn
import torch
import numpy as np
import torch.nn.functional
from torch.distributions import Normal

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose2d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
        if isinstance(module, nn.InstanceNorm2d):
            module.weight = nn.init.constant_(module.weight, 1)
            module.bias = nn.init.constant_(module.bias, 0)

class SE(nn.Module):
    def __init__(self, n_channels):
        super(SE, self).__init__()
        num_hidden = max(n_channels // 16, 4)
        self.se = nn.Sequential(nn.Linear(n_channels, num_hidden), nn.ReLU(inplace=False),
                                nn.Linear(num_hidden, n_channels), nn.Sigmoid())

    def forward(self, x):
        se = torch.mean(x, dim=[2, 3])
        se = se.view(se.size(0), -1)
        se = self.se(se)
        se = se.view(se.size(0), -1, 1, 1)
        return x * se
    

class Upsample(nn.Module):
    def __init__(self, n_channels, n_out, size=None, scale_factor=None,  mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size
        self.conv_1 = nn.Conv2d(n_channels, n_out, 3, stride=1, padding=1, bias=False, dilation=1)

    def forward(self, x):
        x = nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)
        return self.conv_1(x)

class BlockEncoder(nn.Module):
    """BN + Swish + Conv (3x3x3) + BN + Swish + Conv (3x3x3) + SE"""

    def __init__(self, n_channels, residual=True, with_se=True):
        super(BlockEncoder, self).__init__()

        self.residual = residual
        self.with_se = with_se

        self.bn_0 = nn.InstanceNorm2d(n_channels, eps=1e-5, momentum=0.05)
        self.act_0 = nn.SiLU()
        self.conv_0 = nn.Conv2d(n_channels, n_channels, 3, stride=1, padding=1, bias=True, dilation=1)
        self.bn_1 = nn.InstanceNorm2d(n_channels, eps=1e-5, momentum=0.05)
        self.act_1 = nn.SiLU()
        self.conv_1 = nn.Conv2d(n_channels, n_channels, 3, stride=1, padding=1, bias=True, dilation=1)
        self.se = SE(n_channels)
        
        self.output_channels = n_channels

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): of size (B, C_in, H, W, D)
        """
        out = self.bn_0(x)
        out = self.act_0(out)
        out = self.conv_0(out)
        out = self.bn_1(out)
        out = self.act_1(out)
        out = self.conv_1(out)
        if self.with_se:
            out = self.se(out)
        if self.residual:
            return out + x
        else:
            return out


class BlockDecoder(nn.Module):
    def __init__(self, n_channels, ex=6, residual=True, with_se=True):
        super(BlockDecoder, self).__init__()

        self.residual = residual
        self.with_se = with_se

        hidden_dim = int(round(n_channels * ex))
        
        

        self.bn_0 = nn.InstanceNorm2d(n_channels, eps=1e-5, momentum=0.05)
        self.conv_0 = nn.Conv2d(n_channels , hidden_dim, 3, stride=1, padding=1, bias=True, dilation=1)
        self.bn_1 = nn.InstanceNorm2d(hidden_dim, eps=1e-5, momentum=0.05)
        self.act_1 = nn.SiLU()
        self.dw_conv_1 = nn.Conv2d(hidden_dim, hidden_dim, 5, stride=1, padding=2, bias=True, dilation=1, groups=hidden_dim)
        self.bn_2 = nn.InstanceNorm2d(hidden_dim, eps=1e-5, momentum=0.05)
        self.act_2 = nn.SiLU()
        self.se_2 = SE(hidden_dim)
        self.conv_2 = nn.Conv2d(hidden_dim, n_channels, 1, stride=1, padding=0, bias=True, dilation=1)

        self.output_channels = n_channels        

    def forward(self, x):
        out = self.bn_0(x)
        out = self.conv_0(out)
        out = self.bn_1(out)
        out = self.act_1(out)
        out = self.dw_conv_1(out)
        out = self.bn_2(out)
        out = self.act_2(out)
        if self.with_se:
            out = self.se_2(out)
        out = self.conv_2(out)
        if self.residual:
            return out + x
        else:
            return out

class BlockFinal(nn.Module):
    def __init__(self, n_channels, n_out, residual=True, with_se=True, stride=1, mean_final=False):
        super(BlockFinal, self).__init__()

        self.residual = residual
        self.with_se = with_se
        self.stride = stride
        self.mean_final = mean_final

        self.bn_0 = nn.InstanceNorm2d(n_channels, eps=1e-5, momentum=0.05)
        self.act_0 = nn.SiLU()
        self.conv_0 = nn.Conv2d(n_channels, n_channels, 3, stride=stride, padding=1, bias=True, dilation=1)

        self.bn_1 = nn.InstanceNorm2d(n_channels, eps=1e-5, momentum=0.05)
        self.act_1 = nn.SiLU()
        self.conv_1 = nn.Conv2d(n_channels, n_channels, 3, stride=stride, padding=1, bias=True, dilation=1)
        self.se = SE(n_channels)

        self.bn_2 = nn.InstanceNorm2d(n_channels, eps=1e-5, momentum=0.05)
        self.act_2 = nn.SiLU()
        self.conv_2 = nn.Conv2d(n_channels, n_out, 1, stride=1, padding=0, bias=True, dilation=1)
        

    def forward(self, x):
        out = self.bn_0(x)
        out = self.act_0(out)
        out = self.conv_0(out)

        out = self.bn_1(out)
        out = self.act_1(out)
        out = self.conv_1(out)

        if self.with_se:
            out = self.se(out)
        if self.residual:
            out = out + x

        out = self.bn_2(out)
        out = self.act_2(out)
        out = self.conv_2(out)
        if self.mean_final:
            out = torch.mean(out, [2,3], keepdim=True)

        return out 


class BlockQ(nn.Module):
    def __init__(self, n_channels, n_hidden, n_out, **kwargs):
        super().__init__()

        self.bn_0 = nn.InstanceNorm2d(n_channels, eps=1e-5, momentum=0.05)
        self.act_0 = nn.SiLU()
        self.conv_0 = nn.Conv2d(n_channels, n_hidden, 3, stride=1, padding=1, bias=True, dilation=1)
        self.bn_1 = nn.InstanceNorm2d(n_hidden, eps=1e-5, momentum=0.05)
        self.act_1 = nn.SiLU()
        self.conv_1 = nn.Conv2d(n_hidden, n_hidden, 3, stride=1, padding=1, bias=True, dilation=1)

        self.se = SE(n_hidden)
        self.last_conv = nn.utils.weight_norm(nn.Conv2d(n_hidden, n_out, 1, 1, 0, 1, 1, False))

    def forward(self, x):
        
        out = self.bn_0(x)
        out = self.act_0(out)
        out = self.conv_0(out)
        
        out = self.bn_1(out)
        out = self.act_1(out)
        out = self.conv_1(out)
        

        out = self.se(out)
        
        x = self.last_conv(out)
        return x


class AsFeatureMap_up(nn.Module):
    def __init__(self, input_dim, target_shape, weightnorm=True, **kwargs):
        super().__init__()

        self._input_dim = input_dim
        out_features = np.prod(target_shape)
        self.linear = nn.Linear(input_dim, out_features)
        self.linear = nn.utils.weight_norm(self.linear, dim=0, name="weight")
        self._output_shp = target_shape



    def forward(self, x):
        batch_size = x.size()[0]
        x = self.linear(x)
        return x.view([batch_size,]+self._output_shp)


class AsFeatureMap_down(nn.Module):
    def __init__(self, input_shape, target_dim, weightnorm=True, **kwargs):
        super().__init__()

        self._input_shp = input_shape
        input_features = np.prod(input_shape)
        self.linear = nn.Linear(input_features, target_dim)
        self.linear = nn.utils.weight_norm(self.linear, dim=0, name="weight")



    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x
    
class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class BlockFinalImg(nn.Module):
    def __init__(self, n_channels: int=3, n_out: int=64, last_act: str='tanh') -> None:
        super().__init__()

        self.res = nn.Sequential(
            ResnetBlock(n_channels, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True),
            ResnetBlock(n_channels, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True),
            ResnetBlock(n_channels, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True),
            ResnetBlock(n_channels, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True),
            ResnetBlock(n_channels, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True),
            ResnetBlock(n_channels, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True),
        )


    
        self.convt2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=7, padding=3, padding_mode='reflect')
        self.norm2 = nn.InstanceNorm2d(n_channels)
        self.act = nn.LeakyReLU(0.2, True)
        self.convt3 = nn.Conv2d(in_channels=n_channels, out_channels=n_out, kernel_size=7, padding=3, padding_mode='reflect')
        if last_act=='tanh':
            self.act_last = nn.Tanh()
        else:
            self.act_last = nn.Identity()
    
    def forward(self, x):
        out = self.res(x)

        out = self.convt2(out)
        out = self.norm2(out)
        out = self.act(out)
        out = self.convt3(out)
        out = self.act_last(out)
        return out




def soft_clamp(x: torch.Tensor, v: int=10):
    return x.div(v).tanh_().mul(v)

def soft_clamp_img(x: torch.Tensor):
    return (x.div(5).tanh_() + 1 ) / 2 