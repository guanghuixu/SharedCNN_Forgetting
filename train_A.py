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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
epochs = 200
batch_size = 256
p_frenquent = 50
model_name = 'A'
test_n = 1
best_acc = -1000

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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck')


# net = Net_5().to(device=device)
net = Net_10().to(device=device)
# net = cifar_resnet20().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=True)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200,eta_min=0.001)

def test(net, testloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
    return 100.0 * correct / total

# train
for epoch in range(epochs):  # loop over the dataset multiple times
    net.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.to(device))
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

    if epoch % test_n == 0 or epoch == epochs - 1:
        final_acc = test(net, testloader)
        if final_acc > best_acc:
            best_acc = final_acc
            best_epoch = epoch
            best_dict = net.state_dict()
        p_str = 'Epoch: {} | Acc: {}% | best_epoch: {} | best_acc: {}%'.format(
            epoch, final_acc, best_epoch, best_acc)
        with open('./logs/A_log.txt', 'a+') as f:
            f.writelines(p_str + '\n')
        print(p_str)
print('Finished Training')

PATH = './checkpoints/cifar_net_{}.pth'.format(model_name)
torch.save(best_dict, PATH)

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    p_str = 'Accuracy of {} : {}%'.format(classes[i], 100.0 * class_correct[i] / class_total[i])
    with open('./logs/A_log.txt', 'a+') as f:
        f.writelines(p_str + '\n')
    print(p_str)


