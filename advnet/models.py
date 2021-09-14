import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=5):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        kernel = gkern(kernel_size, 2).astype(np.float32)
        print('kernel size is {0}.'.format(self.kernel_size))
        assert self.kernel_size % 2 == 1, 'kernel size must be odd.'
        self.kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=self.kernel, requires_grad=False)

    def forward(self, x):
        x1 = x[:, 0, :, :].unsqueeze_(1)
        x2 = x[:, 1, :, :].unsqueeze_(1)
        x3 = x[:, 2, :, :].unsqueeze_(1)
        padding = self.kernel_size // 2
        x1 = F.conv2d(x1, self.weight, padding=padding)
        x2 = F.conv2d(x2, self.weight, padding=padding)
        x3 = F.conv2d(x3, self.weight, padding=padding)
        x = torch.cat([x1, x2, x3], dim=1)
        return x

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        backbone = resnet.resnet18(pretrained=True)
        self.convnet = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
        )
        self.fc = nn.Sequential(nn.Linear(512, 9216), nn.PReLU(), nn.Linear(9216, 4096), nn.PReLU())

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

# #################################################
# ## SiameseNet
# #################################################
# class SiameseNet(nn.Module):
#     def __init__(self, embedding_net):
#         super(SiameseNet, self).__init__()
#         self.embedding_net = embedding_net
#
#     def forward(self, x1, x2):
#         output1 = self.embedding_net(x1)
#         output2 = self.embedding_net(x2)
#         return output1, output2
#
# #################################################
# ## TripletNet
# #################################################
# class TripletNet(nn.Module):
#     def __init__(self, embedding_net):
#         super(TripletNet, self).__init__()
#         self.embedding_net = embedding_net
#
#     def forward(self, x1, x2, x3):
#         output1 = self.embedding_net(x1)
#         output2 = self.embedding_net(x2)
#         output3 = self.embedding_net(x3)
#         return output1, output2, output3

#################################################
## NoiseGenerator
#################################################

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size, 1, padding, groups=groups, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(out_planes),)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Encoder(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(Encoder, self).__init__()
        self.block1 = BasicBlock(in_planes, out_planes, kernel_size, stride, padding, groups, bias)
        self.block2 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        # TODO bias=True
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride, padding, output_padding, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(out_planes),
                                nn.ReLU(inplace=True),)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)
        return x


class NoiseGenerator(nn.Module):
    """
    Generate Model Architecture
    """
    def __init__(self):
        """
        Model initialization
        :type x_n: int
        """
        super(NoiseGenerator, self).__init__()

        base = resnet.resnet18(pretrained=True)

        self.in_block = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )
        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4
        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)
        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, 3, 2, 2, 0)
        self.tanh = nn.Tanh()


    def forward(self, x):
        n, c, h_old, w_old = x.shape
        h_new = h_old
        w_new = w_old
        # Transform to 32 times
        if h_old % 32 != 0:
            h_new = h_old - h_old % 32
        if w_old % 32 != 0:
            w_new = w_old - w_old % 32
        x = F.interpolate(x, (h_new, w_new), mode="bilinear", align_corners=True)
        # Initial block
        x = self.in_block(x)
        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        # Decoder blocks
        #d4 = e3 + self.decoder4(e4)
        d4 = e3 + self.decoder4(e4)
        d3 = e2 + self.decoder3(d4)
        d2 = e1 + self.decoder2(d3)
        d1 = x + self.decoder1(d2)
        # Classifier
        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)
        y = self.tanh(y)
        y = F.interpolate(y, (h_old, w_old), mode="bilinear", align_corners=True)
        return y

def get_gaussian_blur(kernel_size, device):
    gaussian_blur = GaussianBlur(kernel_size)
    return gaussian_blur.to(device)

if __name__ == '__main__':
    # x1 = torch.randn((4,3,112,112))
    # siamese = EmbeddingNet()
    # y = siamese(x1)
    # print(y.shape)

    # y = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
    # print(y.pow(2))
    # print(y.pow(2).sum(1, keepdim=True))
    # y /= y.pow(2).sum(1, keepdim=True).sqrt()
    # print(y)

    x1= torch.randn((4,3,112,112))
    ngnet = NoiseGenerator()
    y = ngnet(x1)
    print(y.shape)
