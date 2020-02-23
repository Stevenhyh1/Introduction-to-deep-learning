import torch
import torch.nn as nn

def conv3x3(in_channel, out_channel, stride, padding):
    return nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=stride,padding=padding,bias=False)

def conv1x1(in_channel, out_channel, stride):
    return nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=stride,bias=False)

class Block(nn.Module):
    def __init__(self, in_channel, channels, stride=1, padding=0, norm_layer=None):
        super(Block,self).__init__()
        if len(channels)!=3:
            raise ValueError('ResNet101 must have 3 layers in each convolution block')
        self.conv1=conv1x1(in_channel,channels[0],stride)
        self.bn1=norm_layer(channels[0])
        self.relu=nn.ReLU(inplace=True)
        self.conv2=conv3x3(channels[0],channels[1],stride,padding)
        self.bn2=norm_layer(channels[1])
        self.conv3=conv1x1(channels[1],channels[2],stride)
        self.bn3=norm_layer(channels[2])
        self.stride=stride
    
    def forward(self,x):
        identity=x
        res=self.conv1(x)
        res=self.bn1(res)
        res=self.relu(res)
        res=self.conv2(res)
        res=self.bn2(res)
        res=self.relu(res)
        res=self.conv3(res)
        res=self.bn3(res)
        res+=identity
        res=self.relu(res)
        return res

class Classification(nn.Module):
    def __init__(self,input_dimension,output_dimension):
        super(Classification, self).__init__()

    def forward(self,x):
        return x

