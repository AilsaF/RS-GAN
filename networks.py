from torch.nn import init
import torch
import torch.nn as nn
from torch.nn import utils
import torch.nn.functional as F
import math

from gen_resblocks import Block as GenBlock
from disc_resblocks import Block as DiscBlock
from disc_resblocks import OptimizedBlock


# ====== SN DCGAN for CIFAR, STL =======
class DCGenerator32(nn.Module):
    # initializers
    def __init__(self, dim_z=128, num_features=64, channel=3, first_kernel=4):
        super(DCGenerator32, self).__init__()
        self.dim_z = dim_z
        self.num_features = num_features
        self.first_kernel = first_kernel
        self.l1 = nn.Linear(dim_z, num_features*8*first_kernel*first_kernel)
        self.deconv1 = nn.ConvTranspose2d(num_features*8, num_features*4, 4, 2, 1)
        self.deconv1_bn = nn.BatchNorm2d(num_features*4)
        self.deconv2 = nn.ConvTranspose2d(num_features*4, num_features*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(num_features*2)
        self.deconv3 = nn.ConvTranspose2d(num_features*2, num_features, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(num_features)
        self.conv4 = nn.Conv2d(num_features, channel, 3, 1, 1)
    
    def forward(self, input):
        x = self.l1(input)
        x = x.view(-1, self.num_features * 8, self.first_kernel, self.first_kernel)
        for i in range(1, 4):
            x = getattr(self, 'deconv{}'.format(i))(x)
            x = getattr(self, 'deconv{}_bn'.format(i))(x)
            x = F.relu(x)
        x = F.tanh(self.conv4(x))
        return x


class SNDCDiscriminator32(nn.Module):
    # initializers
    def __init__(self, num_features=64, channel=3, first_kernel=4):
        super(SNDCDiscriminator32, self).__init__()
        self.num_features = num_features
        self.first_kernel = first_kernel
        self.conv1 = utils.spectral_norm(nn.Conv2d(channel, num_features, 3, 1, 1))
        self.conv2 = utils.spectral_norm(nn.Conv2d(num_features, num_features, 4, 2, 1))
        self.conv3 = utils.spectral_norm(nn.Conv2d(num_features, num_features * 2, 3, 1, 1))
        self.conv4 = utils.spectral_norm(nn.Conv2d(num_features * 2, num_features * 2, 4, 2, 1))
        self.conv5 = utils.spectral_norm(nn.Conv2d(num_features * 2, num_features * 4, 3, 1, 1))
        self.conv6 = utils.spectral_norm(nn.Conv2d(num_features * 4, num_features * 4, 4, 2, 1))
        self.conv7 = utils.spectral_norm(nn.Conv2d(num_features * 4, num_features * 8, 3, 1, 1))
        self.proj = utils.spectral_norm(nn.Linear(num_features * 8 * first_kernel * first_kernel, 1, bias=False))

    def forward(self, input):
        x = input
        for i in range(1, 8):
            x = getattr(self, 'conv{}'.format(i))(x)
            x = F.leaky_relu(x, 0.1)
        x = x.view(x.size(0), -1)
        y = self.proj(x)
        return y

 
class DCGenerator256(nn.Module):
    def __init__(self, dim_z=128, G_h_size=32, image_size=256, n_channels=3):
        super(DCGenerator256, self).__init__()
        model = []
        mult = image_size // 8

        # start block
        model.append(nn.ConvTranspose2d(dim_z, G_h_size * mult, kernel_size=4, stride=1, padding=0, bias=False))
        model.append(nn.BatchNorm2d(G_h_size * mult))
        model.append(nn.ReLU())

        # middel block
        while mult > 1:
            model.append(nn.ConvTranspose2d(G_h_size * mult, G_h_size * (mult//2), kernel_size=4, stride=2, padding=1, bias=False))
            model.append(nn.BatchNorm2d(G_h_size * (mult//2)))
            model.append(nn.ReLU())
            mult = mult // 2

        # end block
        model.append(nn.ConvTranspose2d(G_h_size, n_channels, kernel_size=4, stride=2, padding=1, bias=False))
        model.append(nn.Tanh())
        self.model = nn.Sequential(*model)

    def forward(self, input):
        input = input.view(input.shape[0], input.shape[1], 1, 1)
        output = self.model(input)
        return output


class SNDCDiscriminator256(nn.Module):
    def __init__(self, D_h_size=32, image_size=256, n_channels=3):
        super(SNDCDiscriminator256, self).__init__()
        self.D_h_size = D_h_size
        model = []
        # start block
        model.append(utils.spectral_norm(nn.Conv2d(n_channels, D_h_size, kernel_size=4, stride=2, padding=1, bias=False)))
        model.append(nn.LeakyReLU(0.1, inplace=True))
        image_size_new = image_size // 2

        # middle block
        mult = 1
        while image_size_new > 4:
            model.append(utils.spectral_norm(nn.Conv2d(D_h_size * mult, D_h_size * (2*mult), kernel_size=4, stride=2, padding=1, bias=False)))
            model.append(nn.LeakyReLU(0.1, inplace=True))
            image_size_new = image_size_new // 2
            mult *= 2

        self.model = nn.Sequential(*model)
        self.mult = mult

        # end block
        in_size  = int(D_h_size * mult * 4 * 4)
        self.proj = utils.spectral_norm(nn.Linear(in_size, 1, bias=False))

    def forward(self, input):
        y = self.model(input)
        y = y.view(-1, self.D_h_size * self.mult * 4 * 4)
        output = self.proj(y)
        return output


# =======  define ResNet SNGAN structure for CIFAR ===========
class ResNetGenerator32(nn.Module):
    """Generator generates 32x32."""
    def __init__(self, num_features=256, dim_z=128, channel=3, bottom_width=4,
                 activation=F.relu, num_classes=0, bottleneck=False):
        super(ResNetGenerator32, self).__init__()
        self.num_features = num_features
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.activation = activation
        self.num_classes = num_classes

        self.l1 = nn.Linear(dim_z, num_features * bottom_width ** 2) 

        self.block2 = GenBlock(num_features, num_features,
                            activation=activation, upsample=True,
                            num_classes=num_classes, bottleneck=bottleneck) 
        self.block3 = GenBlock(num_features, num_features,
                            activation=activation, upsample=True,
                            num_classes=num_classes, bottleneck=bottleneck)  
        self.block4 = GenBlock(num_features, num_features,
                            activation=activation, upsample=True,
                            num_classes=num_classes, bottleneck=bottleneck) 
        self.b5 = nn.BatchNorm2d(num_features)
        self.conv5 = nn.Conv2d(num_features, channel, 1, 1)

    def forward(self, z, y=None, **kwargs):
        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        for i in range(2, 5):
            h = getattr(self, 'block{}'.format(i))(h, y, **kwargs)
        h = self.activation(self.b5(h))
        return torch.tanh(self.conv5(h))


class SNResNetProjectionDiscriminator32(nn.Module):
    def __init__(self, num_features=256, channel=3, num_classes=0, activation=F.relu, bottleneck=False):
        super(SNResNetProjectionDiscriminator32, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = OptimizedBlock(channel, num_features, activation=activation, bottleneck=bottleneck)
        self.block2 = DiscBlock(num_features, num_features,
                            activation=activation, downsample=True, bottleneck=bottleneck)
        self.block3 = DiscBlock(num_features, num_features,
                            activation=activation, downsample=True, bottleneck=bottleneck)
        self.block4 = DiscBlock(num_features, num_features,
                            activation=activation, downsample=True, bottleneck=bottleneck)
        self.proj = utils.spectral_norm(nn.Linear(num_features, 1, bias=False))
        self._initialize()

    def _initialize(self):
        init.orthogonal_(self.proj.weight.data)

    def forward(self, x, y=None):
        h = x
        for i in range(1, 5):
            h = getattr(self, 'block{}'.format(i))(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.proj(h)
        return output


# ========  define SNGAN structure for STL ========
class ResNetGenerator48(nn.Module):
    """Generator generates 48x48."""
    def __init__(self, num_features=64, dim_z=128, bottom_width=6,
                 activation=F.relu, num_classes=0, bottleneck=False):
        super(ResNetGenerator48, self).__init__()
        self.num_features = num_features
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.activation = activation
        self.num_classes = num_classes

        self.l1 = nn.Linear(dim_z, 8 * num_features * bottom_width ** 2) 
        self.block2 = GenBlock(num_features * 8, num_features * 4,
                            activation=activation, upsample=True,
                            num_classes=num_classes, bottleneck=bottleneck)  
        self.block3 = GenBlock(num_features * 4, num_features * 2,
                            activation=activation, upsample=True,
                            num_classes=num_classes, bottleneck=bottleneck)  
        self.block4 = GenBlock(num_features * 2, num_features,
                            activation=activation, upsample=True,
                            num_classes=num_classes, bottleneck=bottleneck)  
        self.b5 = nn.BatchNorm2d(num_features)
        self.conv5 = nn.Conv2d(num_features, 3, 1, 1)  

    def forward(self, z, y=None, **kwargs):
        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        for i in range(2, 5):
            h = getattr(self, 'block{}'.format(i))(h, y, **kwargs)
        h = self.activation(self.b5(h))
        return F.tanh(self.conv5(h))


class SNResNetProjectionDiscriminator48(nn.Module):
    def __init__(self, num_features=64, num_classes=0, activation=F.relu, bottleneck=False):
        super(SNResNetProjectionDiscriminator48, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = OptimizedBlock(3, num_features, activation=activation, bottleneck=bottleneck)
        self.block2 = DiscBlock(num_features, num_features * 2,
                            activation=activation, downsample=True, bottleneck=bottleneck)
        self.block3 = DiscBlock(num_features * 2, num_features * 4,
                            activation=activation, downsample=True, bottleneck=bottleneck)
        self.block4 = DiscBlock(num_features * 4, num_features * 8,
                            activation=activation, downsample=True, bottleneck=bottleneck)
        self.block5 = DiscBlock(num_features * 8, num_features * 16,
                                activation=activation, downsample=True, bottleneck=bottleneck)
        self.proj = utils.spectral_norm(nn.Linear(num_features*16, 1, bias=False))
        self._initialize()

    def _initialize(self):
        init.orthogonal_(self.proj.weight.data)

    def forward(self, x, y=None):
        h = x
        for i in range(1, 6):
            h = getattr(self, 'block{}'.format(i))(h)
        h = self.activation(h)
        # Global pooling
        h = h.sum(dim=(2, 3))
        output = self.proj(h)
        return output

def init_ortho_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        init.orthogonal_(m.weight)

def init_xavierunif_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        init.xavier_uniform_(m.weight)

def getGD_SN(structure, dataset, image_size, num_features, ignoreD=False, dim_z=128, bottleneck=False):
    leaky_relu = lambda x: F.leaky_relu(x, negative_slope=0.1)
    if structure == 'resnet':
        if image_size == 32:
            netG = ResNetGenerator32(num_features=num_features, bottleneck=bottleneck)
            if not ignoreD:
                netD = SNResNetProjectionDiscriminator32(num_features=num_features//2, activation=leaky_relu, bottleneck=bottleneck)
        elif image_size == 48:
            netG = ResNetGenerator48(num_features=num_features, bottleneck=bottleneck)
            if not ignoreD:
                netD = SNResNetProjectionDiscriminator48(num_features=num_features, activation=leaky_relu, bottleneck=bottleneck)

    if structure == 'dcgan':
        if image_size == 32:
            netG = DCGenerator32(num_features=num_features, dim_z=dim_z)
            if not ignoreD:
                netD = SNDCDiscriminator32(num_features=num_features)
        elif image_size == 48:
            netG = DCGenerator32(num_features=num_features, first_kernel=6)
            if not ignoreD:
                netD = SNDCDiscriminator32(num_features=num_features, first_kernel=6)
        elif image_size == 256:
            netG = DCGenerator256(dim_z=dim_z, G_h_size=num_features)
            if not ignoreD:
                netD = SNDCDiscriminator256(D_h_size=num_features)
        netG.apply(init_xavierunif_weights)
        netD.apply(init_ortho_weights)

    if ignoreD:  # if eval FID / IS
        netD = None
    return netG, netD
