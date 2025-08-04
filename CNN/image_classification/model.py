import torch.nn as nn

class ImageClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding="same")
        self.batchnorm1 = nn.BatchNorm2d(num_features=8)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, stride=1, padding="same")
        self.batchnorm2 = nn.BatchNorm2d(num_features=32)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=1568, out_features=600)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(600, 10)


    def forward(self, x):
        
        #convolution pass
        x = self.cnn1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.cnn2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        #linear pass
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


        