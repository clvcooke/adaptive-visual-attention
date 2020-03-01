import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms


def get_random_shift(scale=2, image_size=28):
    pad_amount = image_size * (scale - 1)
    crop_size = image_size + pad_amount
    random_shift = transforms.Compose([
        transforms.Pad(pad_amount, fill=0),
        transforms.RandomCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    return random_shift


def get_train_valid_loader(task, batch_size, random_seed, valid_size=0.2):
    assert ((valid_size >= 0) and (valid_size <= 1)), "[!] valid_size should be in the range [0, 1]."
    assert task.upper() == 'MNIST', "only MNIST is supported"
    random_shift = get_random_shift(scale=1)
    data_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=random_shift),
        batch_size=batch_size, shuffle=True)
    all_indices = np.arange(0, data_loader.dataset.data.shape[0])
    np.random.seed(random_seed)
    np.random.shuffle(all_indices)
    train_amount = (int(len(all_indices) * (1 - valid_size)) // batch_size) * batch_size
    train_sampler = SubsetRandomSampler(all_indices[:train_amount])
    valid_sampler = SubsetRandomSampler(all_indices[train_amount:])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=random_shift),
        batch_size=batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=random_shift),
        batch_size=batch_size, sampler=valid_sampler)
    classes = 10
    return train_loader, valid_loader, classes


def get_test_loader(task, batch_size):
    if task.upper() != 'MNIST':
        raise RuntimeError()
    random_shift = get_random_shift()
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, download=True,
                       transform=random_shift),
        batch_size=batch_size, shuffle=True)
    return test_loader
