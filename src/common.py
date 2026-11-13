"""
Shared code for Assignment 3: data loading, the quantization-ready ResNet-18,
training / evaluation helpers and small measurement utilities.
"""
import os
import time
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.ao.quantization import QuantStub, DeQuantStub, fuse_modules, fuse_modules_qat

# torch.ao.quantization (eager mode) and torch.jit print deprecation warnings on
# recent PyTorch versions; they do not affect results, so hide them for clean logs.
warnings.filterwarnings("ignore")

# All paths are relative to the project root (the folder that contains src/),
# so the scripts work no matter which directory they are started from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
RESULT_DIR = os.path.join(ROOT, "results")
ONNX_DIR = os.path.join(ROOT, "onnx_models")
LOG_DIR = os.path.join(ROOT, "logs")


def find_cifar_root():
    """Folder that contains 'cifar-10-batches-py'. An existing copy is reused
    (torchvision then verifies the files and skips the download). Order:
    $CIFAR_ROOT, data/cifar10, data. If none exists, data/cifar10 is used and
    the dataset is downloaded there."""
    candidates = [os.environ.get("CIFAR_ROOT"),
                  os.path.join(ROOT, "data", "cifar10"),
                  os.path.join(ROOT, "data")]
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "cifar-10-batches-py")):
            return c
    return os.path.join(ROOT, "data", "cifar10")


DATA_DIR = find_cifar_root()

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")

# Checkpoint paths (single place so every script agrees)
FP32_PATH = os.path.join(MODEL_DIR, "fp32_resnet18.pth")
PTQ_PATH = os.path.join(MODEL_DIR, "ptq_int8_resnet18.pt")          # TorchScript
QAT_PATH = os.path.join(MODEL_DIR, "qat_int8_resnet18.pt")          # TorchScript
UNSTRUCT_PATH = os.path.join(MODEL_DIR, "unstructured_pruned_resnet18.pth")
UNSTRUCT_SPARSE_PATH = os.path.join(MODEL_DIR, "unstructured_pruned_resnet18_sparse.pth")
STRUCT_PATH = os.path.join(MODEL_DIR, "structured_pruned_resnet18.pth")
STRUCT_PTQ_PATH = os.path.join(MODEL_DIR, "structured_pruned_ptq_int8_resnet18.pt")  # bonus: pruning + INT8


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs():
    for d in (MODEL_DIR, RESULT_DIR, ONNX_DIR):
        os.makedirs(d, exist_ok=True)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def get_transforms(train):
    norm = T.Normalize(CIFAR_MEAN, CIFAR_STD)
    if train:
        return T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                          T.ToTensor(), norm])
    return T.Compose([T.ToTensor(), norm])


def get_datasets(fake=False):
    """Returns (train_set, train_set_no_aug, test_set).
    fake=True uses random images so the pipeline can be smoke-tested offline."""
    if fake:
        tr = torchvision.datasets.FakeData(1000, (3, 32, 32), 10, transform=get_transforms(True))
        tr_plain = torchvision.datasets.FakeData(1000, (3, 32, 32), 10, transform=get_transforms(False))
        te = torchvision.datasets.FakeData(500, (3, 32, 32), 10, transform=get_transforms(False),
                                           random_offset=1000)
        return tr, tr_plain, te
    tr = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=get_transforms(True))
    tr_plain = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=get_transforms(False))
    te = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=get_transforms(False))
    return tr, tr_plain, te


def get_loaders(batch_size=128, fake=False, workers=None):
    tr, _, te = get_datasets(fake)
    if workers is None:
        workers = min(8, os.cpu_count() or 2)
    pin = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(tr, batch_size, shuffle=True,
                                               num_workers=workers, pin_memory=pin,
                                               persistent_workers=workers > 0)
    test_loader = torch.utils.data.DataLoader(te, 256, shuffle=False, num_workers=workers,
                                              pin_memory=pin, persistent_workers=workers > 0)
    return train_loader, test_loader


def get_calibration_loader(num_images=2000, batch_size=64, fake=False, seed=0):
    """Random subset of the TRAINING set (no augmentation) used for PTQ calibration."""
    _, tr_plain, _ = get_datasets(fake)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(tr_plain), generator=g)[:min(num_images, len(tr_plain))].tolist()
    subset = torch.utils.data.Subset(tr_plain, idx)
    return torch.utils.data.DataLoader(subset, batch_size, shuffle=False, num_workers=2)


# ----------------------------------------------------------------------------
# Model: ResNet-18 adapted to 32x32 CIFAR images and ready for eager-mode
# quantization (QuantStub/DeQuantStub, separate ReLU modules, FloatFunctional
# for the residual add so it can be quantized).
# ----------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, mid_planes=None):
        super().__init__()
        mid = mid_planes or planes          # mid_planes < planes after channel pruning
        self.conv1 = nn.Conv2d(in_planes, mid, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(mid, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes))
        self.add = nn.quantized.FloatFunctional()   # quantizable residual add

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.add.add(out, self.shortcut(x))
        return self.relu2(out)


class ResNet18(nn.Module):
    """channel_cfg: optional list of 8 ints = mid channels of each BasicBlock
    (used to rebuild a structurally pruned network)."""
    stage_planes = (64, 128, 256, 512)

    def __init__(self, num_classes=10, channel_cfg=None):
        super().__init__()
        self.channel_cfg = list(channel_cfg) if channel_cfg else None
        cfg = iter(self.channel_cfg) if self.channel_cfg else None
        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        in_planes = 64
        layers = []
        for i, planes in enumerate(self.stage_planes):
            stride = 1 if i == 0 else 2
            blocks = []
            for s in (stride, 1):
                mid = next(cfg) if cfg else None
                blocks.append(BasicBlock(in_planes, planes, s, mid))
                in_planes = planes
            layers.append(nn.Sequential(*blocks))
        self.layer1, self.layer2, self.layer3, self.layer4 = layers
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.avgpool(x), 1)
        x = self.fc(x)
        return self.dequant(x)

    def blocks(self):
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            for b in layer:
                yield b

    def fuse_model(self, qat=False):
        """Fuse Conv+BN(+ReLU) groups. qat=True keeps BN trainable (fuse_modules_qat)."""
        fuse = fuse_modules_qat if qat else fuse_modules
        fuse(self, [["conv1", "bn1", "relu"]], inplace=True)
        for b in self.blocks():
            fuse(b, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True)
            if len(b.shortcut) > 0:
                fuse(b.shortcut, [["0", "1"]], inplace=True)


def load_fp32_model(path=FP32_PATH, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model = ResNet18(channel_cfg=ckpt.get("channel_cfg"))
    model.load_state_dict(ckpt["state_dict"])
    return model


def save_fp32_model(model, path, **extra):
    torch.save({"state_dict": model.state_dict(),
                "channel_cfg": getattr(model, "channel_cfg", None), **extra}, path)


# ----------------------------------------------------------------------------
# Training / evaluation
# ----------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, scheduler=None, log_every=100):
    model.train()
    crit = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
        if log_every and (i + 1) % log_every == 0:
            print(f"    batch {i + 1}/{len(loader)}  loss {loss_sum / total:.4f}  acc {100 * correct / total:.2f}%")
    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device="cpu", return_preds=False):
    model.eval()
    correct, total, preds = 0, 0, []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = model(x).argmax(1)
        correct += (p == y).sum().item()
        total += y.size(0)
        if return_preds:
            preds.append(p.cpu())
    acc = 100.0 * correct / total
    if return_preds:
        return acc, torch.cat(preds).numpy()
    return acc


def make_optimizer(model, lr, epochs, steps_per_epoch, wd=5e-4):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs,
                                                steps_per_epoch=steps_per_epoch, pct_start=0.25)
    return opt, sched


def finetune(model, train_loader, test_loader, device, epochs, lr, tag=""):
    """Short fine-tuning loop used after pruning. Fixed schedule; the weights
    after the LAST epoch are kept (no checkpoint selection on the test set).
    Test accuracy is printed per epoch for monitoring only."""
    if epochs <= 0:
        return evaluate(model, test_loader, device)
    model.to(device)
    opt, sched = make_optimizer(model, lr, epochs, len(train_loader))
    acc = None
    for ep in range(epochs):
        loss, tr_acc = train_one_epoch(model, train_loader, opt, device, sched, log_every=0)
        acc = evaluate(model, test_loader, device)
        print(f"  [{tag}] fine-tune epoch {ep + 1}/{epochs}  loss {loss:.4f}  train {tr_acc:.2f}%  test {acc:.2f}%")
    return acc


# ----------------------------------------------------------------------------
# Measurement helpers
# ----------------------------------------------------------------------------
def file_size_mb(path):
    return os.path.getsize(path) / (1024 ** 2)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum((p != 0).sum().item() for p in model.parameters())
    return total, nonzero


@torch.no_grad()
def count_macs(model, input_size=(1, 3, 32, 32)):
    """Multiply-accumulate count of Conv2d and Linear layers (float models)."""
    macs = []

    def conv_hook(m, i, o):
        k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        macs.append(o.numel() * k)

    def fc_hook(m, i, o):
        macs.append(m.in_features * m.out_features)

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(fc_hook))
    was_training = model.training
    model.eval()
    model(torch.zeros(input_size))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return sum(macs)


@torch.no_grad()
def measure_latency(fn, x, warmup=10, runs=100):
    """Mean / p95 latency in ms of fn(x) on CPU."""
    for _ in range(warmup):
        fn(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return float(times.mean()), float(np.percentile(times, 95))
