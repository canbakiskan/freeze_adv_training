import torch
import torch.nn as nn
from torchvision import datasets, transforms

import numpy as np
from os import path


def cifar10(args):

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose([transforms.ToTensor(), ])

    trainset = datasets.CIFAR10(
        root=args.directory + "data/original_dataset",
        train=True,
        download=False,
        transform=transform_train,
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2
    )

    testset = datasets.CIFAR10(
        root=args.directory + "data/original_dataset",
        train=False,
        download=False,
        transform=transform_test,
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


def cifar10_blackbox(args):

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    # Read
    if args.attack_otherbox_type == "B-T":
        filename = "B-T.npy"
    elif args.attack_otherbox_type == "PW-T":
        filename = "PW-T.npy"
    test_blackbox = np.load(
        args.directory + "data/attacked_dataset/" + filename)

    cifar10 = datasets.CIFAR10(
        path.join(args.directory, "data/original_dataset"),
        train=False,
        transform=None,
        target_transform=None,
        download=False,
    )

    tensor_x = torch.Tensor(test_blackbox / np.max(test_blackbox))
    tensor_y = torch.Tensor(cifar10.targets).long()

    tensor_data = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    attack_loader = torch.utils.data.DataLoader(
        tensor_data, batch_size=args.test_batch_size, shuffle=False, **kwargs
    )

    return attack_loader
