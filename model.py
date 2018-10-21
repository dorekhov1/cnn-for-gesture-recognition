'''
    Write a model for gesture classification.
'''

import torch as torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, model, channels):
        super(Net, self).__init__()

        self.model = model

        if model == 0:
            self.conv1 = torch.nn.Conv1d(channels, 64, kernel_size=3, stride=1, padding=1)
            self.pool = torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
            self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
            self.conv3 = torch.nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
            self.fc1 = torch.nn.Linear(3200, 128)
            self.fc2 = torch.nn.Linear(128, 26)

        elif model == 1:
            self.conv1 = torch.nn.Conv1d(channels, 32, kernel_size=3, stride=1, padding=1)
            self.pool = torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
            self.conv2 = torch.nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
            self.conv3 = torch.nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1)
            self.fc1 = torch.nn.Linear(1600, 128)
            self.fc2 = torch.nn.Linear(128, 26)

        elif model == 2:
            self.conv1 = nn.Conv1d(channels, 100, kernel_size=10, stride=1, padding=0)
            self.conv2 = nn.Conv1d(100, 200, kernel_size=5, stride=1, padding=0)
            self.conv3 = nn.Conv1d(200, 50, kernel_size=3, stride=1, padding=0)

            self.maxpool = nn.MaxPool1d(kernel_size=5, stride=5, padding=0)

            self.conv4 = nn.Conv1d(50, 100, kernel_size=6, stride=1, padding=0)
            self.conv5 = nn.Conv1d(100, 150, kernel_size=11, stride=1, padding=0)

            self.avgpool = nn.AvgPool1d(kernel_size=2, stride=1, padding=0)

            self.activation = nn.LeakyReLU(0.1)
            self.sigmoid = nn.Sigmoid()

            self.fc1 = torch.nn.Linear(150, 50)
            self.fc2 = torch.nn.Linear(50, 26)

        elif model == 3:
            self.conv1 = torch.nn.Conv1d(channels, 32, kernel_size=3, stride=1, padding=1)
            self.pool = torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
            self.conv2 = torch.nn.Conv1d(32, 64, kernel_size=6, stride=1, padding=1)
            self.conv3 = torch.nn.Conv1d(64, 32, kernel_size=12, stride=1, padding=1)
            self.fc1 = torch.nn.Linear(1408, 128)
            self.fc2 = torch.nn.Linear(128, 26)

        elif model == 4:

            self.conv1 = nn.Conv1d(channels, 32, 3)
            self.conv2 = nn.Conv1d(32, 64, 5)
            self.conv3 = nn.Conv1d(64, 128, 7)
            self.conv4 = nn.Conv1d(128, 256, 9)
            self.conv5 = nn.Conv1d(256, 128, 7)
            self.conv6 = nn.Conv1d(128, 64, 9)

            self.fc1 = nn.Linear(4224, 26)
            self.l_relu = nn.LeakyReLU(0.1)
            self.sigmoid = nn.Sigmoid()

        elif model == 5:
            self.conv1 = torch.nn.Conv1d(channels, 32, kernel_size=3, stride=1, padding=1)
            self.pool = torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
            self.conv2 = torch.nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
            self.conv3 = torch.nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
            self.conv4 = torch.nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
            self.fc1 = torch.nn.Linear(1600, 128)
            self.fc2 = torch.nn.Linear(128, 26)

        elif model == 7:
            self.down1 = torch.nn.Conv1d(channels, 32, kernel_size=3, stride=1, padding=1)
            self.down2 = torch.nn.Conv1d(32, 128, kernel_size=6, stride=1, padding=1)
            self.down4 = torch.nn.Conv1d(128, 256, kernel_size=12, stride=1, padding=1)
            self.up1 = torch.nn.Conv1d(256, 64, kernel_size=24, stride=1, padding=1)
            self.up3 = torch.nn.Conv1d(64, 32, kernel_size=48, stride=1, padding=1)
            self.fc1 = torch.nn.Linear(704, 26)

        elif model == 8:
            self.down1 = torch.nn.Conv1d(channels, 32, kernel_size=3, stride=1, padding=1)
            self.down2 = torch.nn.Conv1d(32, 64, kernel_size=6, stride=1, padding=1)
            self.down3 = torch.nn.Conv1d(64, 128, kernel_size=12, stride=1, padding=1)
            self.down4 = torch.nn.Conv1d(128, 256, kernel_size=12, stride=1, padding=1)
            self.up1 = torch.nn.Conv1d(256, 64, kernel_size=24, stride=1, padding=1)
            self.up2 = torch.nn.Conv1d(64, 32, kernel_size=12, stride=1, padding=1)
            self.fc1 = torch.nn.Linear(1568, 26)

        elif model == 9:
            self.down1 = torch.nn.Conv1d(channels, 32, kernel_size=3, stride=1, padding=1)
            self.down2 = torch.nn.Conv1d(32, 128, kernel_size=6, stride=1, padding=1)
            self.down4 = torch.nn.Conv1d(128, 256, kernel_size=12, stride=1, padding=1)
            self.up1 = torch.nn.Conv1d(256, 64, kernel_size=24, stride=1, padding=1)
            self.up3 = torch.nn.Conv1d(64, 32, kernel_size=48, stride=1, padding=1)
            self.fc1 = torch.nn.Linear(704, 26)

        elif model == 10:
            self.down1 = torch.nn.Conv1d(channels, 32, kernel_size=3, stride=1, padding=1)
            self.down2 = torch.nn.Conv1d(32, 64, kernel_size=6, stride=1, padding=1)
            self.down3 = torch.nn.Conv1d(64, 128, kernel_size=12, stride=1, padding=1)
            self.down4 = torch.nn.Conv1d(128, 256, kernel_size=12, stride=1, padding=1)
            self.up1 = torch.nn.Conv1d(256, 64, kernel_size=24, stride=1, padding=1)
            self.up2 = torch.nn.Conv1d(64, 32, kernel_size=12, stride=1, padding=1)
            self.fc1 = torch.nn.Linear(1568, 26)

        elif model == 11:
            self.down1 = torch.nn.Conv1d(channels, 32, kernel_size=3)
            self.down2 = torch.nn.Conv1d(32, 64, kernel_size=6)
            self.down3 = torch.nn.Conv1d(64, 128, kernel_size=12)
            self.down4 = torch.nn.Conv1d(128, 256, kernel_size=24)
            self.up1 = torch.nn.Conv1d(256, 200, kernel_size=24)
            self.up2 = torch.nn.Conv1d(200, 125, kernel_size=12)
            self.up3 = torch.nn.Conv1d(125, 80, kernel_size=6)
            self.up4 = torch.nn.Conv1d(80, 50, kernel_size=3)
            self.fc1 = torch.nn.Linear(900, 450)
            self.fc2 = torch.nn.Linear(450, 200)
            self.fc3 = torch.nn.Linear(200, 26)

        elif model == 12:
            self.down1 = torch.nn.Conv1d(channels, 32, kernel_size=3)
            self.down2 = torch.nn.Conv1d(32, 64, kernel_size=6)
            self.down3 = torch.nn.Conv1d(64, 128, kernel_size=12)
            self.down4 = torch.nn.Conv1d(128, 256, kernel_size=12)
            self.up1 = torch.nn.Conv1d(256, 128, kernel_size=24)
            self.up2 = torch.nn.Conv1d(128, 64, kernel_size=12)
            self.up3 = torch.nn.Conv1d(64, 32, kernel_size=6)
            self.fc1 = torch.nn.Linear(1024, 256)
            self.fc2 = torch.nn.Linear(256, 26)

    def forward(self, x):

        if self.model == 0:

            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(x.shape[0], -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)

        elif self.model == 1:

            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(x.shape[0], -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)

        elif self.model == 2:

            x = self.activation(self.conv1(x))
            x = self.activation(self.conv2(x))
            x = self.activation(self.conv3(x))
            x = self.maxpool(x)
            x = self.activation(self.conv4(x))
            x = self.activation(self.conv5(x))
            x = self.avgpool(x)
            x = x.view(x.shape[0], -1)
            x = self.activation(self.fc1(x))
            x = self.activation(self.fc2(x))
            x = self.sigmoid(x)

        elif self.model == 3:

            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(x.shape[0], -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)

        elif self.model == 4:

            x = self.conv1(x)
            x = self.l_relu(x)
            x = self.conv2(x)
            x = self.l_relu(x)
            x = self.conv3(x)
            x = self.l_relu(x)
            x = self.conv4(x)
            x = self.l_relu(x)
            x = self.conv5(x)
            x = self.l_relu(x)
            x = self.conv6(x)
            x = self.l_relu(x)

            x = x.view(x.shape[0], -1)
            x = self.fc1(x)
            x = self.sigmoid(x)

        elif self.model == 5:

            x = F.leaky_relu(self.conv1(x))
            x = F.leaky_relu(self.conv2(x))
            x = F.leaky_relu(self.conv3(x))
            x = F.leaky_relu(self.conv4(x))
            x = self.pool(x)
            x = x.view(x.shape[0], -1)
            x = F.tanh(self.fc1(x))
            x = self.fc2(x)

        elif self.model == 7:
            x = F.leaky_relu(self.down1(x))
            x = F.leaky_relu(self.down2(x))
            x = F.leaky_relu(self.down4(x))
            x = F.leaky_relu(self.up1(x))
            x = F.leaky_relu(self.up3(x))
            x = x.view(x.shape[0], -1)
            x = F.leaky_relu(self.fc1(x))

        elif self.model == 8:
            x = F.leaky_relu(self.down1(x), 0.25)
            x = F.leaky_relu(self.down2(x), 0.25)
            x = F.leaky_relu(self.down3(x), 0.25)
            x = F.leaky_relu(self.down4(x), 0.25)
            x = F.leaky_relu(self.up1(x), 0.25)
            x = F.leaky_relu(self.up2(x), 0.25)
            x = x.view(x.shape[0], -1)
            x = F.leaky_relu(self.fc1(x), 0.1)

        elif self.model == 10:
            x = F.leaky_relu(self.down1(x), 0.25)
            x = F.leaky_relu(self.down2(x), 0.25)
            x = F.leaky_relu(self.down3(x), 0.25)
            x = F.leaky_relu(self.down4(x), 0.25)
            x = F.leaky_relu(self.up1(x), 0.25)
            x = F.leaky_relu(self.up2(x), 0.25)
            x = x.view(x.shape[0], -1)
            x = F.leaky_relu(self.fc1(x), 0.1)

        elif self.model == 11:
            x = F.elu(self.down1(x), 0.1)
            x = F.elu(self.down2(x), 0.1)
            x = F.elu(self.down3(x), 0.1)
            x = F.elu(self.down4(x), 0.1)
            x = F.elu(self.up1(x), 0.1)
            x = F.elu(self.up2(x), 0.1)
            x = F.elu(self.up3(x), 0.1)
            x = F.elu(self.up4(x), 0.1)
            x = x.view(x.shape[0], -1)
            x = F.elu(self.fc1(x), 0.1)
            x = F.elu(self.fc2(x), 0.1)
            x = F.elu(self.fc3(x), 0.1)

        elif self.model == 12:
            x = F.leaky_relu(self.down1(x), 0.2)
            x = F.leaky_relu(self.down2(x), 0.2)
            x = F.leaky_relu(self.down3(x), 0.2)
            x = F.leaky_relu(self.down4(x), 0.2)
            x = F.leaky_relu(self.up1(x), 0.2)
            x = F.leaky_relu(self.up2(x), 0.2)
            x = F.leaky_relu(self.up3(x), 0.2)
            x = x.view(x.shape[0], -1)
            x = F.leaky_relu(self.fc1(x), 0.1)
            x = F.leaky_relu(self.fc2(x), 0.1)

        return x
