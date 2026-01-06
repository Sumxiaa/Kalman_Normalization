"""
CIFAR-10 Training Example with Kalman Normalization

This example demonstrates how to use KalmanNorm and GroupKalmanNorm
with a simple ResNet-style network on CIFAR-100.

Usage:
    python examples/cifar10_train.py --norm_type gkn --num_groups 4 --p_rate 0.9
    python examples/cifar10_train.py --norm_type gn  # Baseline with BatchNorm
"""

"""
CIFAR-10 Training Example with Kalman Normalization

This example demonstrates how to use KalmanNorm and GroupKalmanNorm
with a simple ResNet-style network on CIFAR-10.
"""

import argparse
import sys
import os
import csv  # <-- 新增：用于保存 CSV

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from kalman_norm import KalmanNorm
from group_kalman_norm import GroupKalmanNorm


class BasicBlock(nn.Module):
    """Basic residual block with configurable normalization"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 norm_type: str = 'bn',
                 split_num: int = 1,
                 num_groups: int = 4,
                 p_rate: float = 0.9):
        super().__init__()
        self.norm_type = norm_type

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)

        if norm_type == 'bn':
            self.norm1 = nn.BatchNorm2d(out_channels)
        elif norm_type == 'kn':
            self.norm1 = KalmanNorm(out_channels, split_num, p_rate)
        elif norm_type == 'gkn':
            self.norm1 = GroupKalmanNorm(out_channels, num_groups, p_rate)
        elif norm_type == 'gn':
            self.norm1 = nn.GroupNorm(num_groups, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)

        if norm_type == 'bn':
            self.norm2 = nn.BatchNorm2d(out_channels)
        elif norm_type == 'kn':
            self.norm2 = KalmanNorm(out_channels, split_num, p_rate)
        elif norm_type == 'gkn':
            self.norm2 = GroupKalmanNorm(out_channels, num_groups, p_rate)
        elif norm_type == 'gn':
            self.norm2 = nn.GroupNorm(num_groups, out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            if norm_type == 'bn':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            elif norm_type == 'kn':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                    KalmanNorm(out_channels, split_num, p_rate)
                )
            elif norm_type == 'gkn':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                    GroupKalmanNorm(out_channels, num_groups, p_rate)
                )
            elif norm_type == 'gn':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                    nn.GroupNorm(num_groups, out_channels)
                )

    def forward(self, x, pre_mean=None, pre_var=None):
        out = self.conv1(x)

        if self.norm_type in ['kn', 'gkn']:
            out, m1, v1 = self.norm1(out, pre_mean, pre_var)
        else:
            out = self.norm1(out)
            m1, v1 = None, None

        out = F.relu(out)
        out = self.conv2(out)

        if self.norm_type in ['kn', 'gkn']:
            out, m2, v2 = self.norm2(out, m1, v1)
        else:
            out = self.norm2(out)
            m2, v2 = None, None

        shortcut = x
        for layer in self.shortcut:
            if isinstance(layer, (KalmanNorm, GroupKalmanNorm)):
                shortcut, _, _ = layer(shortcut, None, None)
            else:
                shortcut = layer(shortcut)

        out = out + shortcut
        out = F.relu(out)

        return out, m2, v2


class SimpleResNet(nn.Module):
    """Simple ResNet for CIFAR-10 with configurable normalization"""

    def __init__(self,
                 num_classes: int = 100,
                 norm_type: str = 'bn',
                 split_num: int = 1,
                 num_groups: int = 4,
                 p_rate: float = 0.9):
        super().__init__()
        self.norm_type = norm_type

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)

        if norm_type == 'bn':
            self.norm1 = nn.BatchNorm2d(16)
        elif norm_type == 'kn':
            self.norm1 = KalmanNorm(16, split_num, p_rate)
        elif norm_type == 'gkn':
            self.norm1 = GroupKalmanNorm(16, num_groups, p_rate)
        elif norm_type == 'gn':
            self.norm1 = nn.GroupNorm(num_groups, 16)

        self.layer1 = self._make_layer(16, 16, 2, 1, norm_type, split_num, num_groups, p_rate)
        self.layer2 = self._make_layer(16, 32, 2, 2, norm_type, split_num, num_groups, p_rate)
        self.layer3 = self._make_layer(32, 64, 2, 2, norm_type, split_num, num_groups, p_rate)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, in_ch, out_ch, num_blocks, stride, norm_type, split_num, num_groups, p_rate):
        layers = []
        layers.append(BasicBlock(in_ch, out_ch, stride, norm_type, split_num, num_groups, p_rate))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_ch, out_ch, 1, norm_type, split_num, num_groups, p_rate))
        return nn.ModuleList(layers)

    def forward(self, x):
        x = self.conv1(x)

        if self.norm_type in ['kn', 'gkn']:
            x, m, v = self.norm1(x, None, None)
        else:
            x = self.norm1(x)
            m, v = None, None

        x = F.relu(x)

        for block in self.layer1:
            x, m, v = block(x, m, v)

        for block in self.layer2:
            x, m, v = block(x, m, v)

        for block in self.layer3:
            x, m, v = block(x, m, v)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    return total_loss / len(train_loader), 100. * correct / total


def test_epoch(model, test_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return total_loss / len(test_loader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description='CIFAR-100 Training with Kalman Normalization')
    parser.add_argument('--norm_type', type=str, default='kn', choices=['gkn', 'gn'])
    parser.add_argument('--split_num', type=int, default=4)
    parser.add_argument('--num_groups', type=int, default=4)
    parser.add_argument('--p_rate', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'Using device: {device}')

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    print('Loading CIFAR-100 dataset...')
    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    if args.norm_type == 'kn':
        while args.batch_size % args.split_num != 0:
            args.batch_size -= 1
        print(f'Adjusted batch size to {args.batch_size}')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=True)

    model = SimpleResNet(
        norm_type=args.norm_type,
        split_num=args.split_num,
        num_groups=args.num_groups,
        p_rate=args.p_rate
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {num_params:,}')

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # CSV 文件初始化
    csv_path = "training_log_cifar100.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc"])

    print(f'\nStarting training for {args.epochs} epochs...')
    print('=' * 60)

    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}')
        print('-' * 40)

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = test_epoch(model, test_loader, device)
        scheduler.step()

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test  Loss: {test_loss:.4f}, Test  Acc: {test_acc:.2f}%')

        # 写入 CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, test_loss, test_acc])

    print('\n' + '=' * 60)
    print('Training completed!')
    print(f'CSV saved to: {csv_path}')


if __name__ == '__main__':
    main()
