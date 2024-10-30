import torch.nn as nn
from torchvision.models import resnet18

class MyModel(nn.Module):
    def __init__(self, n, m):
        super(MyModel, self).__init__()
        
        # Define the model
        self.attack_model = resnet18(pretrained=False)
        
        self.attack_model.conv1 = nn.Conv2d(in_channels=1,
            out_channels=64,
            kernel_size=2,
            stride=1,
            padding=3,
            bias=False
        )
        
        self.attack_model.fc = nn.Linear(512, 2*(m**2) + n**2)
        
        # self.attack_model = nn.Sequential(
        #     nn.Conv2d(in_channels=1,out_channels=16,kernel_size=2,stride=1,padding=0),
        #     nn.LeakyReLU(),
        #     # nn.MaxPool2d(kernel_size=1,stride=1),
        #     nn.Flatten(),
        #     nn.Linear(16*2*7,out_features=(2*(m**2) + n**2))
        # )
        
    def forward(self, data):
        
        out = self.attack_model(data)
        return out
        