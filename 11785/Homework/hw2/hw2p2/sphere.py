import torch
import torch.nn as nn
import torchvision.models as models
import math

from torch.nn import Parameter
from torch.autograd import Variable

def myphi(x,m):
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos())
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta,self.m)
            phi_theta = phi_theta.clamp(-1*self.m,1)

        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)
        output = (cos_theta,phi_theta)
        return output # size=(B,Classnum,2)

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
    def __init__(self, in_channel, num_class, hiddens=[3,4,6,3], task='Classification', use_feature = False):
        super(ResNet,self).__init__()

        self.task = task
        self.use_feature = use_feature
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
        self.fc = nn.Linear(4*512, 512)
        self.sp = AngleLinear(512, num_class)

    
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
        x = torch.flatten(x,1)
        x = self.fc(x)

        # if self.task == 'Verification':
        #     return x
        if self.use_feature:
            return x

        x = self.sp(x)

        return x
    
    def change_task(self,task_name):
        
        self.task = task_name






if __name__ == "__main__":
    input = torch.rand(8, 3, 30, 30)
    model = ResNet(3, 2600, [3,4,6,3])
    # model = Sphere20(2600)
    output= model(input)
    print(output.shape)

