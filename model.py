from config import Config
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet(nn.Module):
    def __init__(self, pretrained=True, postprocess=False):
        super(ResNet, self).__init__()
        self.postprocess = postprocess
        self.net = models.resnet18(pretrained=pretrained)
        #self.net =models.mobilenet_v2(pretrained=pretrained) 
        self.net.fc = nn.Linear(in_features=512, out_features=Config.get_class_num()) # 2048 for resnet50; 512,resnet18

    def forward(self, x):
        x = self.net(x)
        if self.postprocess:
            x = F.softmax(x, dim=1)
        return x
