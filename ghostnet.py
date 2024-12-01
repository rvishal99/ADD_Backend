import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import math
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve

# Define MelSpectrogramDataset, GhostModule, GhostBottleneck, and GhostNet without Dropout

class MelSpectrogramDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.labels = pd.read_csv(csv_file, header=None).iloc[:, 0].tolist()
        self.image_files = sorted(os.listdir(root_dir))
        self.transform = transform

    def _len_(self):
        return len(self.image_files)

    def _getitem_(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        label = int(self.labels[idx])  # 0 for fake, 1 for real

        if self.transform:
            image = self.transform(image)

        return image, label


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se=False):
        super(GhostBottleneck, self).__init__()
        self.stride = stride

        # Pointwise GhostModule
        self.ghost1 = GhostModule(inp, hidden_dim, kernel_size=1, relu=True)

        # Depthwise convolution
        if self.stride > 1:
            self.depthwise = nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=hidden_dim, bias=False
            )
            self.bn = nn.BatchNorm2d(hidden_dim)
        else:
            self.depthwise = None

        # Squeeze-and-excitation (optional)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=1, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, hidden_dim, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
        ) if use_se else None

        # Second pointwise GhostModule
        self.ghost2 = GhostModule(hidden_dim, oup, kernel_size=1, relu=False)

        # Shortcut
        self.shortcut = nn.Sequential()
        if stride > 1 or inp != oup:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        residual = self.shortcut(x)

        x = self.ghost1(x)
        if self.depthwise is not None:
            x = self.depthwise(x)
            x = self.bn(x)
        if self.se is not None:
            x = x * self.se(x)
        x = self.ghost2(x)

        return x + residual


class GhostNet(nn.Module):
    def __init__(self, num_classes=2, width=1.0):
        super(GhostNet, self).__init__()
        cfgs = [
            [3, 16, 16, 0, 1],
            [3, 48, 24, 0, 2],
            [3, 72, 24, 0, 1],
            [5, 72, 40, 1, 2],
            [5, 120, 40, 1, 1],
            [3, 240, 80, 0, 2],
            [3, 200, 80, 0, 1],
            [3, 184, 80, 0, 1],
            [3, 184, 80, 0, 1],
            [3, 480, 112, 1, 1],
            [3, 672, 112, 1, 1],
            [5, 672, 160, 1, 2],
            [5, 960, 160, 0, 1],
            [5, 960, 160, 1, 1],
            [5, 960, 160, 1, 1],
        ]

        # Build first layer
        output_channel = 16
        self.conv_stem = nn.Conv2d(3, output_channel, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)

        input_channel = output_channel
        stages = []
        for k, exp_size, c, use_se, s in cfgs:
            output_channel = c
            hidden_channel = exp_size
            stages.append(GhostBottleneck(input_channel, hidden_channel, output_channel, k, s, use_se))
            input_channel = output_channel
        self.bottlenecks = nn.Sequential(*stages)

        # Build last layer
        output_channel = 960
        self.conv_head = nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.act2 = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.bottlenecks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        return x