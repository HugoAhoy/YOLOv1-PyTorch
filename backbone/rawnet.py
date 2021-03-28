'''
As the architecture shown in Fig 3 of YOLOv1 paper, 
The Conv Layers of YOLO can be devided into 6 parts.
Here the 6 parts are named after 'YOLOConvs{}'.format{i} for i in range(1,7).
'''

import torch
import torch.nn as nn


class YOLOConvs1(nn.Module):
    def __init__(self, in_channel=3, out_channel=64):
        super(YOLOConvs1, self).__init__()
        # ConvLayer:(7,7)kernel, stride=2,inchannel=3, outchannel=64,padding=3
        self.conv = nn.Conv2d(in_channel, out_channel, 7, stride=2, padding=3)

        # MaxPool:(2,2),stride=2
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.maxpool(self.act(self.conv(x)))
        return out

class YOLOConvs2(nn.Module):
    def __init__(self, in_channel=64, out_channel=192):
        super(YOLOConvs2, self).__init__()
        # ConvLayer:(3,3)kernel, stride=1,inchannel=64, outchannel=192,padding=1
        self.conv = nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1)

        # MaxPool:(2,2),stride=2
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.maxpool(self.act(self.conv(x)))
        return out

class YOLOConvs3(nn.Module):
    def __init__(self, in_channel=192, out_channel=512):
        super(YOLOConvs3, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 128, 1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(256,out_channel, 3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    
    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(x1))
        x3 = self.act(self.conv3(x2))
        x4 = self.act(self.conv1(x3))
        out = self.maxpool(x4)
        return out

class YOLOConvs4(nn.Module):
    def __init__(self, in_channel=512, out_channel=1024):
        super(YOLOConvs4, self).__init__()
        self.convsequence = nn.Sequential()
        for i in range(4):
            self.convsequence.add_module(nn.Conv2d(in_channel, 256, 1, stride=1, padding=0))
            self.convsequence.add_module(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            self.convsequence.add_module(nn.Conv2d(256, in_channel, 3, stride=1, padding=1))
            self.convsequence.add_module(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.conv1 = nn.Conv2d(in_channel, 512, 1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(512, out_channel, 3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(2,stride=2)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    
    def forward(self, x):
        x1 = self.convsequence(x)
        x2 = self.act(self.conv1(x1))
        x3 = self.act(self.conv2(x2))
        out = self.maxpool(x3)
        return out

class YOLOConvs5(nn.Module):
    def __init__(self, in_channel=1024, out_channel=1024):
        super(YOLOConvs5, self).__init__()
        self.convsequence = nn.Sequential()
        for i in range(2):
            self.convsequence.add_module(nn.Conv2d(in_channel, 512, 1, stride=1, padding=0))
            self.convsequence.add_module(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            self.convsequence.add_module(nn.Conv2d(512, in_channel, 3, stride=1, padding=1))
            self.convsequence.add_module(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        
        self.conv1 = nn.Conv2d(in_channel, 1024, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(1024, out_channel, 3, stride=2, padding=1)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x1 = self.convsequence(x)
        x2 = self.act(self.conv1(x1))
        out = self.act(self.conv2(x2))
        return out

class YOLOConvs6(nn.Module):
    def __init__(self, in_channel=1024, out_channel=1024):
        super(YOLOConvs6, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 1024, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(1024, out_channel, 3, stride=1, padding=1)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    
    def forward(self, x):
        x1 = self.act(self.conv1(x))
        out = self.act(self.conv2(x1))
        return out

class YOLOv1(nn.Module):
    def __init__(self):
        super(YOLOv1, self).__init__()
        self.Convs1 = YOLOConvs1()
        self.Convs2 = YOLOConvs2()
        self.Convs3 = YOLOConvs3()
        self.Convs4 = YOLOConvs4()
        self.Convs5 = YOLOConvs5()
        self.Convs6 = YOLOConvs6()

        self.fc1 = nn.Linear(7*7*1024, 4096)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc2 = nn.Linear(4096, 7*7*30)
    
    def forward(self, x):
        x1 = self.Convs1(x)
        x2 = self.Convs2(x1)
        x3 = self.Convs3(x2)
        x4 = self.Convs4(x3)
        x5 = self.Convs5(x4)
        x6 = self.Convs6(x5)

        flattened = x6.view(-1)

        x7 = self.fc2(self.act(self.fc1(flattened)))
        out = x7.reshape(7,7,30)
        return out