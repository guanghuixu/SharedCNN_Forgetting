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
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
p_frenquent = 500
model_name = 'B'

parser = argparse.ArgumentParser()
parser.add_argument("--share_n", type=int, default=1)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=256)
args = parser.parse_args()

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

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck')
parameters_list = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5',
                    'fc1', 'fc2', 'fc3', 'fc4', 'fc5']

A = Net_10().to(device)
A_dict = torch.load('./checkpoints/cifar_net_A.pth', map_location={'cuda:6':'cuda'})
A.load_state_dict(A_dict)

B = Net_10().to(device=device)
print(B.conv1.conv.weight[0, 0, :10])
B_dict = B.state_dict()
map_parameters_list = parameters_list[-args.share_n:]
for key in B_dict.keys():
    key_ = key.split('.')[0]
    if key_ in map_parameters_list:
        B_dict[key] = A_dict[key]    # initialize B with A shared layers, the first 10-share_n layer
        print(key)
B.load_state_dict(B_dict)
print(B.conv1.conv.weight[0, 0, :10])

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(B.parameters(), lr=0.1, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=True)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200,eta_min=0.001)

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

    new_acc = 100.0 * correct / total
    return new_acc

# train
for epoch in range(args.epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    B.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = B(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        scheduler.step()

        # print statistics
        running_loss += loss.item()
        if i and i % p_frenquent == 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / p_frenquent))
            running_loss = 0.0
    B.eval()
    new_acc = test(A, B, testloader, args.share_n)
    p_str = 'Share_n: {} | Epoch: {}/{} | new_acc: {}%'.format(args.share_n, epoch, args.epochs, new_acc)
    print(p_str)
    with open('./logs/B{}_log.txt'.format(args.share_n), 'a+') as f:
        f.writelines(p_str + '\n')

# test: the final acc of B
B.eval()
new_acc = test(B, B, testloader, args.share_n)
p_str = 'Share_n: {} | Epoch: {}/{} | B_acc: {}%'.format(args.share_n, epoch, args.epochs, new_acc)
print(p_str)
with open('./logs/B{}_log.txt'.format(args.share_n), 'a+') as f:
    f.writelines(p_str + '\n')

PATH = './checkpoints/cifar_net_B{}.pth'.format(model_name, args.share_n)
torch.save(B.state_dict(), PATH)
print('Finished Training && Save {}'.format(PATH))