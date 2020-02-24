import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

class Bottleneck(nn.Module):
    def __init__(self,in_channels, out_channels, downsample=None):
        super(Bottleneck,self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.relu=nn.ReLU(inplace=True)
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels,out_channels=4*out_channels,kernel_size=1,bias=False),
            nn.BatchNorm2d(4*out_channels)
        )
        self.downsample=downsample

    def forward(self, x):
        identity = x
        out = self.conv(x)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        out += identity
        out = self.relu(out)

        return out

class ResLayer(nn.Module):
    def __init__(self, in_channel, out_channel, num_layer, stride=1):
        super(ResLayer, self).__init__()

        self.conv = nn.ModuleList()
        self.downsample=nn.Sequential(
            nn.Conv2d(in_channel,4*in_channel,kernel_size=1,stride=stride,bias=False),
            nn.BatchNorm2d(4*in_channel)
        )
        self.conv.append(Bottleneck(in_channel,out_channel,downsample=self.downsample))
        for _ in range(num_layer-1):
            self.conv.append(Bottleneck(4*out_channel,out_channel,downsample=None))
    
    def forward(self,x):
        
        return self.conv(x)

class ResNet(nn.Module):
    def __init__(self, in_channel, num_class, hiddens=[3,4,23,3]):
        super(ResNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.layer1 = ResLayer(in_channel, 64, hiddens[0], stride=1)
        self.layer2 = ResLayer(4*64, 128, hiddens[1], stride=2)
        self.layer3 = ResLayer(4*128, 256, hiddens[2], stride=2)
        self.layer4 = ResLayer(4*256, 512, hiddens[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(4*512, num_class)
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)

        return x

class Classification(nn.Module):
    def __init__(self,input_dimension,output_dimension):
        super(Classification, self).__init__()

    def forward(self,x):
        return x



if __name__ == "__main__":
    # model = Bottleneck(64,64,downsample)
    # model = ResLayer(512,256,23,2)
    # model=models.resnet101(3,1000)
    input = torch.rand(5, 3, 224, 224)
    model = ResNet(3, 1000, [3,4,23,3])
    # print(model)
    output = model(input)
    print(output.shape)

