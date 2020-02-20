import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.resnet import cifar_resnet20
from models.ConvFc import Net_5, Net_10

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
print(device)
epochs = 200
batch_size = 256
p_frenquent = 50

## dataloader
transform = transforms.Compose([
    transforms.RandomCrop(size=32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], # mean=[0.5071, 0.4865, 0.4409] for cifar100
        std=[0.2023, 0.1994, 0.2010],  # std=[0.2009, 0.1984, 0.2023] for cifar100
    ),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck')

A = Net_10().to(device)
B = Net_10().to(device)

A.load_state_dict(torch.load('cifar_net_A.pth'))
B.load_state_dict(torch.load('cifar_net_B.pth'))


# test
# net = Net().to(device=device)
# net.load_state_dict(torch.load(PATH))

def test(A, B, testloader, share_n):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = A.forward_A(images.to(device), 10 - share_n)
            outputs = B.forward_B(outputs, 10 - share_n)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

    # print('Accuracy of the network on the 10000 test images: %d %%' % (
    #     100.0 * correct / total))

    new_acc = 100.0 * correct / total
    return new_acc

for share_n in range(10):
    new_acc = test(A, B, testloader, share_n)
    print('*' * 20)
    print('Share_n: {} | acc: {}%'.format(share_n, new_acc))
