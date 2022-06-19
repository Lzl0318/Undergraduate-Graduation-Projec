import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = self.conv1(x)  # 16*14*14
        x = self.conv2(x)  # 32*7*7
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


# device and batch and learning rate
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
LR = 0.01
# data
train_data = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = datasets.CIFAR10(root='./data', train=False)
test_x = torch.unsqueeze(test_data.train_data, dim=1).type(torch.FloatTensor)[:2000].to(device)/255.
test_y = test_data.test_labels[:2000].to(device)


def Accuracy(pred, test):
    counter = 0
    for i in range(test.size(0)):
        if pred[i] == test[i]:
            counter = counter+1
    return counter/test.size(0)


# train
model = CNN().to(device)

loss_func = nn.CrossEntropyLoss()
for epoch in range(5):
    for step, (x, y) in enumerate(train_loader):
        output = model(x.to(device))
        loss = loss_func(output, y.to(device))
        params = model.state_dict()
        dy_dx = torch.autograd.grad(loss, model.parameters())
        original_dy_dx = list((_.detach().clone() for _ in dy_dx))

        # gaussian noise with specific variance
        x = []
        for i in range(len(original_dy_dx)):
            r = 0.1 * torch.randn_like(original_dy_dx[i])
            x.append(r)

        # adding noise to gradient
        noise_dy_dx = []
        for i in range(len(original_dy_dx)):
            noise_dy_dx.append(original_dy_dx[i] + x[i])

        # using noised gradient update model parameters
        params['conv1.0.weight'] = params['conv1.0.weight'] - LR * noise_dy_dx[0]
        params['conv1.0.bias'] = params['conv1.0.bias'] - LR * noise_dy_dx[1]
        params['conv2.0.weight'] = params['conv2.0.weight'] - LR * noise_dy_dx[2]
        params['conv2.0.bias'] = params['conv2.0.bias'] - LR * noise_dy_dx[3]
        params['out.weight'] = params['out.weight'] - LR * noise_dy_dx[4]
        params['out.bias'] = params['out.bias'] - LR * noise_dy_dx[5]
        
        model.load_state_dict(params)

        # show the train details and test results
        if step % 50 == 0:
            test_output = model(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            acc = Accuracy(pred_y, test_y)
            print('EPOCH:', epoch, '|test loss:%.4f' % loss.cpu().item(),
                  '|test accuracy:%.4f' % acc)
print("cnn finish training")
