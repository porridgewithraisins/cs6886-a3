import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import numpy as np
import random

from datetime import datetime
from functools import partial
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(6886)
np.random.seed(6886)
random.seed(6886)


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        hidden_ch = in_ch * expand_ratio
        self.use_residual = stride == 1 and in_ch == out_ch

        layers = []
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(in_ch, hidden_ch, 1, bias=False),
                    nn.BatchNorm2d(hidden_ch),
                    nn.ReLU6(inplace=True),
                ]
            )
        layers.extend(
            [
                nn.Conv2d(
                    hidden_ch, hidden_ch, 3, stride, 1, groups=hidden_ch, bias=False
                ),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_residual else self.conv(x)


class MobileNetV2_CIFAR(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0):
        super().__init__()
        # first conv: stride changed from 2 to 1 for CIFAR-10
        input_ch = int(32 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, input_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_ch),
            nn.ReLU6(inplace=True),
        )

        # 24-channel stage: stride changed from 2 to 1 for CIFAR-10
        cfgs = [
            [1, 16, 1, 1],
            [6, 24, 2, 1],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        layers = []
        for t, c, n, s in cfgs:
            out_ch = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual(input_ch, out_ch, stride, t))
                input_ch = out_ch
        self.features = nn.Sequential(*layers)

        last_ch = int(1280 * width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(input_ch, last_ch, 1, bias=False),
            nn.BatchNorm2d(last_ch),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(last_ch, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class Cutout(object):
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask).expand_as(img)
        return img * mask


def get_loaders(batch_size=512, val_split=5000, num_workers=8):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2430, 0.2610)),
            Cutout(n_holes=1, length=16),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2430, 0.2610)),
        ]
    )

    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    train_len = len(dataset) - val_split
    train_set, val_set = random_split(dataset, [train_len, val_split])
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def evaluate(model, loader, criterion):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item() * y.size(0)
            _, pred = out.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    train_loader, val_loader, test_loader = get_loaders()

    model = MobileNetV2_CIFAR(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.4, momentum=0.9, weight_decay=4e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    scaler = torch.amp.GradScaler("cuda")
    autocast = partial(torch.amp.autocast, "cuda")

    epochs = 300
    best_val = 0.0
    slog_fd = open("training_log.csv", "w")
    slog_writer = csv.writer(slog_fd)
    slog_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        running_loss, running_correct, total = 0.0, 0, 0

        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast():
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * y.size(0)
            _, pred = out.max(1)
            running_correct += pred.eq(y).sum().item()
            total += y.size(0)
            loop.set_postfix(
                train_acc=100 * running_correct / total, train_loss=running_loss / total
            )

        scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader, criterion)

        train_loss_epoch = running_loss / total
        train_acc_epoch = running_correct / total
        print(
            f"[{now()}] Epoch {epoch + 1}: Val Acc={val_acc * 100:.2f}%, Val Loss={val_loss:.4f}"
        )
        slog_writer.writerow(
            [epoch + 1, train_loss_epoch, train_acc_epoch, val_loss, val_acc]
        )

        if val_acc > best_val:
            torch.save(model.state_dict(), "model.pth")
            best_val = val_acc

    slog_fd.close()

    model.load_state_dict(torch.load("model.pth"))
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"[{now()}] Final Test Acc={test_acc * 100:.2f}%, Test Loss={test_loss:.4f}")


if __name__ == "__main__":
    main()
