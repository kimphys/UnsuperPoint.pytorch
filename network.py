import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):  # https://github.com/digantamisra98/Mish
    def forward(self, x):
        return x * F.softplus(x).tanh()

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, bn=True, activation_type='leaky'):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, 
                                out_channels=out_channels, 
                                kernel_size=kernel_size, 
                                stride=stride,
                                padding=pad)
        if bn:
            self.batchNorm = nn.BatchNorm2d(out_channels, momentum=0.03, eps=1e-4)
        else:
            self.batchNorm = None
        
        self.activation_type = activation_type
        if activation_type == 'leaky':
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation_type == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_type == 'mish':
            self.activation = Mish()
        elif activation_type == 'swish':
            self.activation = Swish()
        else:
            self.activation = None

    def forward(self, x):
        x = self.conv2d(x)

        if self.batchNorm is not None:
            x = self.batchNorm(x)

        if self.activation is not None:
            x = self.activation(x)

        return x

class Backbone(nn.Module):
    def __init__(self, in_channels):
        super(Backbone, self).__init__()

        channels = [in_channels,32,32,64,64,128,128,256,256]

        self.conv2d_1 = ConvLayer(in_channels=channels[0],out_channels=channels[1])
        self.conv2d_2 = ConvLayer(in_channels=channels[1],out_channels=channels[2])

        self.conv2d_3 = ConvLayer(in_channels=channels[2],out_channels=channels[3])
        self.conv2d_4 = ConvLayer(in_channels=channels[3],out_channels=channels[4])

        self.conv2d_5 = ConvLayer(in_channels=channels[4],out_channels=channels[5])
        self.conv2d_6 = ConvLayer(in_channels=channels[5],out_channels=channels[6])

        self.conv2d_7 = ConvLayer(in_channels=channels[6],out_channels=channels[7])
        self.conv2d_8 = ConvLayer(in_channels=channels[7],out_channels=channels[8])

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.maxpool(x)

        x = self.conv2d_3(x)
        x = self.conv2d_4(x)
        x = self.maxpool(x)

        x = self.conv2d_5(x)
        x = self.conv2d_6(x)
        x = self.maxpool(x)

        x = self.conv2d_7(x)
        x = self.conv2d_8(x)

        return x

class Score(nn.Module):
    def __init__(self, in_channels):
        super(Score, self).__init__()
        self.conv_1 = ConvLayer(in_channels=in_channels,out_channels=in_channels)
        self.conv_2 = ConvLayer(in_channels=in_channels,out_channels=1, activation_type='sigmoid')

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

def MappingXY(relative, downsampling):

    # relative -> B 2 H W
    H = relative.shape[2]
    W = relative.shape[3]

    pmap = relative.clone().detach()

    for i in range(H):
        for j in range(W):
            pmap[:,0,i,j] = (i + relative[:,0,i,j]) * downsampling
            pmap[:,1,i,j] = (j + relative[:,1,i,j]) * downsampling

    return pmap

class PositionXY(nn.Module):
    def __init__(self, in_channels):
        super(PositionXY, self).__init__()
        self.conv_1 = ConvLayer(in_channels=in_channels,out_channels=in_channels)
        self.conv_2 = ConvLayer(in_channels=in_channels,out_channels=2, activation_type='sigmoid')

    def forward(self, x, downsampling=8):
        x = self.conv_1(x)
        x = self.conv_2(x)

        pmap = MappingXY(x, downsampling)

        return pmap, x

class Descriptor(nn.Module):
    def __init__(self, in_channels):
        super(Descriptor, self).__init__()
        self.conv_1 = ConvLayer(in_channels=in_channels,out_channels=in_channels)
        self.conv_2 = ConvLayer(in_channels=in_channels,out_channels=in_channels, activation_type='None')

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)

        return x

# Under construction
def interpolation(position, descriptor, h , w):

    interpolate = descriptor

    return interpolate


class Unsuperpoint(nn.Module):
    def __init__(self, in_channels=3, s_channels=256, p_channels=256, d_channels=256):
        super(Unsuperpoint, self).__init__()

        self.backbone = Backbone(in_channels=in_channels)
        self.score = Score(in_channels=s_channels)
        self.position = PositionXY(in_channels=p_channels)
        self.descriptor = Descriptor(in_channels=d_channels)

    def forward_once(self, x):
        h, w = x.shape[2], x.shape[3]
        x = self.backbone(x)

        x1 = self.score(x)
        x2, x2_r = self.position(x)
        x3 = self.descriptor(x)
        x3 = interpolation(x2, x3, h, w)

        return x1, x2_r, x2, x3

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2

if __name__ == "__main__":
    inputs1 = torch.autograd.Variable(torch.rand(3,3,256,256))
    inputs2 = torch.autograd.Variable(torch.rand(3,3,256,256))

    print("Input1: {}, Input2: {}".format(inputs1.shape, inputs2.shape))

    model = Unsuperpoint()

    preds_1, preds_2 = model(inputs1, inputs2)

    print("---For input 1---")
    print("Score: ", preds_1[0].shape)
    print("Relative position: ", preds_1[1].shape)
    print("Mapping position: ", preds_1[2].shape)
    print("Descriptor: ", preds_1[3].shape)
    print("---For input 2---")
    print("Score: ", preds_2[0].shape)
    print("Relative position: ", preds_2[1].shape)
    print("Mapping position: ", preds_2[2].shape)
    print("Descriptor: ", preds_2[3].shape) 