import torch
from torch import nn


class CNNModel(torch.nn.Module):

    def __init__(self, out_dimension, in_channels=1, lr=0.01, l2_normalize_output=False):
        super(CNNModel, self).__init__()

        self.lr = lr
        self.l2_normalize_output = l2_normalize_output
        self.in_channels = in_channels
        self.feature_dimension = out_dimension
        self.dimension = self.feature_dimension

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=8,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(1296, 512),
            nn.PReLU(),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Linear(256, self.feature_dimension),
        )

    def debug_forward(self, x):

        x_tmp = x.clone()
        for layer in self.feature_extractor:
            print(x_tmp.shape)
            x_tmp = layer(x_tmp)

        print(x_tmp.shape)

    def forward(self, x):

        x = self.feature_extractor(x)
        if 'l2_normalize_output' in self.__dict__.keys() and self.l2_normalize_output:
            x = x / torch.norm(x, p=2, dim=1).unsqueeze(1)

        return x