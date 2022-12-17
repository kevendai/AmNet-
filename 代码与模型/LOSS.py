import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torchvision.models as models


# 以VGG16为损失网络，从前3个阶段的最后一层提取特征

class VGG16_relu(torch.nn.Module):
    def __init__(self):
        super(VGG16_relu, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cnn = models.vgg16(pretrained=True)
        # cnn.load_state_dict(torch.load(os.path.join('./models/', 'vgg19-dcbb9e9d.pth')))
        cnn = cnn.to(self.device)
        features = cnn.features
        self.relu1 = torch.nn.Sequential()

        self.relu2 = torch.nn.Sequential()

        self.relu3 = torch.nn.Sequential()

        for x in range(4):
            self.relu1.add_module(str(x), features[x])

        for x in range(4, 9):
            self.relu2.add_module(str(x), features[x])

        for x in range(9, 16):
            self.relu3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1 = self.relu1(x)
        relu2 = self.relu2(relu1)
        relu3 = self.relu3(relu2)

        out = {
            'relu1': relu1,
            'relu2': relu2,
            'relu3': relu3,
        }
        return out


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG16_relu())
        self.criterion = nn.MSELoss()
        self.weight = 0.04
        self.IN = nn.InstanceNorm2d(512, affine=False, track_running_stats=False)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)

    def __call__(self, x, y):
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        x = (x - self.mean.to(x)) / self.std.to(x)
        y = (y - self.mean.to(y)) / self.std.to(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        loss = self.weight * self.criterion(self.IN(x_vgg['relu1']), self.IN(y_vgg['relu1']))
        loss += self.weight * self.criterion(self.IN(x_vgg['relu2']), self.IN(y_vgg['relu2']))
        loss += self.weight * self.criterion(self.IN(x_vgg['relu3']), self.IN(y_vgg['relu3']))

        return loss
