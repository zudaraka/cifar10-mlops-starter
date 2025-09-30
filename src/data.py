import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=128):
    t_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))
    ])
    t_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))
    ])
    trainset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=t_train)
    testset  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=t_test)
    trainset, valset = torch.utils.data.random_split(trainset, [45000, 5000])
    return (
        DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2),
        DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    )
