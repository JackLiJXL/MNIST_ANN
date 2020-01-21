import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from matplotlib import pyplot as plt

#Load Dataset
train_data = datasets.MNIST('', train = True, download = True,
                           transform = transforms.Compose([transforms.ToTensor()]))
test_data = datasets.MNIST('', train = False, download = True,
                           transform = transforms.Compose([transforms.ToTensor()]))

#Shuffle & Load Data into Tensors
train_set = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
test_set = torch.utils.data.DataLoader(test_data, batch_size = 8, shuffle = True)

#Create ANN Class
class ANN(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(28*28, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_3 = nn.Linear(64, 64)
        self.layer_4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = F.log_softmax(self.layer_4(x), dim = 1)
        return x

Model = ANN()

loss_record = []

#Train Model
Optimizer = optim.Adam(Model.parameters(), lr = 0.001)

epochs = 20

for epoch in range(epochs):
    for data in train_set:
        x, y = data
        Model.zero_grad()
        y_hat = Model(x.view(-1, 28*28))
        loss = F.nll_loss(y_hat, y)
        loss_record.append(loss)
        loss.backward()
        Optimizer.step()
    print(loss)
