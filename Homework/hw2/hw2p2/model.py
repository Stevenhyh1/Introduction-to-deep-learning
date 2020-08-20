import torch
import torch.nn as nn
import torchvision.models as models
import math

from torch.nn import Parameter
from torch.autograd import Variable

class Bottleneck(nn.Module):
    def __init__(self,in_channels, out_channels, stride, downsample=None):
        super(Bottleneck,self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.relu=nn.ReLU(inplace=True)
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels,out_channels=4*out_channels,kernel_size=1,bias=False),
            nn.BatchNorm2d(4*out_channels)
        )
        self.downsample=downsample

    def forward(self, x):
        # print(f"Bottleneck id: {x.shape}")
        identity = x
        out = self.conv(x)
        
        # print(f"Bottlenect conv: {out.shape}")
        if self.downsample is not None:
            identity = self.downsample(identity)
        # print(f"Downsample: {identity.shape}")
        out += identity
        out = self.relu(out)

        return out

class ResLayer(nn.Module):
    def __init__(self, in_channel, out_channel, num_layer, stride=1, dimchange = False):
        super(ResLayer, self).__init__()

        self.conv = []
        if dimchange is True:
            self.downsample=nn.Sequential(
                nn.Conv2d(in_channel,4*in_channel,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(4*in_channel)
            )
        else:
            self.downsample=nn.Sequential(
                nn.Conv2d(in_channel,4*out_channel,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(4*out_channel)
            )
        self.conv.append(Bottleneck(in_channel,out_channel,stride=stride,downsample=self.downsample))
        for _ in range(num_layer-1):
            self.conv.append(Bottleneck(4*out_channel,out_channel,stride=1, downsample=None))
        self.convlayers = nn.Sequential(*self.conv)
    
    def forward(self,x):
        
        x = self.convlayers(x)

        return x



class ResNet(nn.Module):
    def __init__(self, in_channel, num_class, hiddens=[3,4,6,3], task='Classification'):
        super(ResNet,self).__init__()

        self.task = task
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.layer1 = ResLayer(64, 64, hiddens[0], stride=1, dimchange=True)
        self.layer2 = ResLayer(4*64, 128, hiddens[1], stride=2)
        self.layer3 = ResLayer(4*128, 256, hiddens[2], stride=2)
        self.layer4 = ResLayer(4*256, 512, hiddens[3], stride=2)

        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(1024)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(4*512, num_class) 
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.bn1(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        feat = torch.flatten(x,1)

        if self.task == 'Verification':
            return feat
        
        x = self.fc(feat)
        if self.task == 'Both':
            return feat, x

        return x
    
    def change_task(self,task_name):
        self.task = task_name

if __name__ == "__main__":
    input = torch.rand(8, 3, 30, 30)
    model = ResNet(3, 2600, [3,4,6,3])
    # model = Sphere20(2600)
    output= model(input)
    print(output.shape)

