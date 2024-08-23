
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.4914, 0.4822, 0.4465],
        [0.2471, 0.2435, 0.2616]
    )
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.4914, 0.4822, 0.4465],
        [0.2471, 0.2435, 0.2616]
    )
])

trainset = CIFAR100(root='./data_100', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=512, shuffle=True, drop_last=True, pin_memory=True)


# testset = CIFAR100(root='./data_100', train=False, download=True, transform=transform_test)

import torch
'''
xs = []
ys = []
for x, y in testset:
    xs.append(x)
    ys.append(y)

xs = torch.stack(xs)
ys = torch.Tensor(ys)

torch.save(xs, 'cifar100_transformed/xs.pt')
torch.save(ys, 'cifar100_transformed/ys.pt')
'''
xs = torch.load('cifar100_transformed/xs.pt')
ys = torch.load('cifar100_transformed/ys.pt')

class cached_imagenet(Dataset):

    def __getitem__(self, index):
        return xs[index], ys[index]

    def __len__(self):
        return len(ys)

testset = cached_imagenet()

testloader = DataLoader(testset, batch_size=1024, shuffle=False, drop_last=True, pin_memory=True)
