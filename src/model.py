import torch.nn as nn
from torchvision.models import alexnet, resnet18

class MyModel(nn.Module):
    def __init__(self, n, m):
        super(MyModel, self).__init__()
        
        # Calculate the output dimension based on n and m
        # output_dim = 2 * (n**2) + m**2
        
        # Define a simple fully connected neural network
        linear_input_dim = n*(n+2*m)
        linear_output_dim = (2 * (n**2) + m**2) # Batchsize * 2 * (m**2) + n**2
         
        self.attack_model = nn.Sequential(
            nn.Linear(linear_input_dim, 128),   # Input layer
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),       # Dropout for regularization
            
            nn.Linear(128, 256),   # Hidden layer 1
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 512),   # Hidden layer 2
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),   # Hidden layer 3
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),   # Hidden layer 4
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, linear_output_dim)   # Output layer
        )

        
        # Resnet18
        # self.attack_model = resnet18(pretrained=False)
        
        # self.attack_model.conv1 = nn.Conv2d(in_channels=1,
        #     out_channels=64,
        #     kernel_size=2,
        #     stride=1,
        #     padding=0,
        #     bias=False
        # )
        
        # self.attack_model.fc = nn.Linear(512, 2*(n**2) + m**2)
        
        # Alexnet
        # self.attack_model = alexnet(pretrained=False)
        
        # self.attack_model.features[0] = nn.Conv2d(in_channels=1,
        #     out_channels=64,
        #     kernel_size=2,
        #     stride=1,
        #     padding=0,
        #     bias=False
        # )
        
        # self.attack_model.classifier[6] = nn.Linear(4096, 2*(n**2) + m**2)
        
        # self.attack_model = nn.Sequential(
        #     nn.Conv2d(in_channels=1,out_channels=16,kernel_size=2,stride=1,padding=0),
        #     nn.LeakyReLU(),
        #     # nn.MaxPool2d(kernel_size=1,stride=1),
        #     nn.Flatten(),
        #     nn.Linear(16*2*7,out_features=(2*(m**2) + n**2))
        # )
        
    def forward(self, data):
        
        linear = True
        if linear:
            out = self.attack_model(data.view(data.size(0),-1))
            return out.reshape(10,22)
        else:
            out = self.attack_model(data)
            return out