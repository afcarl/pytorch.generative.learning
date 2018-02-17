from pathlib import PosixPath
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

root = PosixPath("~/.torch/data").expanduser()


def data_loader(dataset, batch_size):
    _loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    return _loader


def cifar10(batch_size, image_size):
    _set = datasets.CIFAR10(root=str(root / "cifar10"), download=True,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    return data_loader(_set, batch_size)


def fashion_mnist(batch_size, image_size):
    _set = datasets.FashionMNIST(root=str(root / "fashonMNIST"), download=True,
                                 transform=transforms.Compose([
                                     transforms.Resize(image_size),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                 ]))
    return data_loader(_set, batch_size)
