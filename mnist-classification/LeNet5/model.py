
import torch.nn as nn

class LeNet5(nn.Module): # number of paramters: 61,706
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1) # (5*5*1+1)*6 = 153
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1) # (5 * 5 * 6 + 1) * 16 = 2,416
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # (400 + 1) * 120 = 48,120
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84) # (120 + 1) * 84 = 10,164
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10) # (84 + 1) * 10 = 850

    def forward(self, img):

        x = self.conv1(img)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1) # convolution layer와 full connected layer 연결
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        output = self.fc3(x)

        return output


class CustomMLP(nn.Module): # number of paramters: 66,250
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """

    def __init__(self):
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(32 * 32, 58)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(58, 29)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(29, 10)  

    def forward(self, img):
        x = img.view(img.size(0), -1)  
        
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        output = self.fc3(x)
        
        return output
