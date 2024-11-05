import torch.nn as nn
from torchvision.models import alexnet, resnet18

class MyModel(nn.Module):
    def __init__(self, model_type, n, m, p, batch_size):
        super(MyModel, self).__init__()
        
        # Calculate the output dimension based on n and m
        self.conv_output_dim = 2 * (n**2) + m**2
        self.batch_size = batch_size
        self.model_type = model_type
        
        # Define a simple fully connected neural network
        self.linear_input_dim = n*(n+2*m+p+1)
        self.linear_output_dim = (n**2) + 2*(m**2) # Batchsize * 2 * (m**2) + n**2
        
        if self.model_type == "linear":
            drop_rate = 0.01
            
            self.attack_model = nn.Sequential(
                nn.Linear(self.linear_input_dim, 128),   # Input layer
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                
                nn.Linear(128, 256), 
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                
                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                
                nn.Linear(512, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                
                nn.Linear(1024, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                
                nn.Linear(2048, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                
                nn.Linear(128, self.linear_output_dim)   # Output layer
            )

            # Print the number of training parameters in the model
            self.count_params(self.attack_model)
            
        # Resnet18
        if self.model_type == "resnet":
            self.attack_model = resnet18(weights=False)
            
            self.attack_model.conv1 = nn.Conv2d(in_channels=1,
                out_channels=64,
                kernel_size=2,
                stride=1,
                padding=0,
                bias=False
            )
            
            self.attack_model.fc = nn.Linear(512, self.conv_output_dim)
            
            # Print the number of training parameters in the model
            self.count_params(self.attack_model)
        
        # Alexnet
        if self.model_type == "alexnet":
            self.attack_model = alexnet(weights=False)
            
            self.attack_model.features[0] = nn.Conv2d(in_channels=1,
                out_channels=64,
                kernel_size=2,
                stride=1,
                padding=0,
                bias=False
            )
        
            self.attack_model.classifier[6] = nn.Linear(4096, self.conv_output_dim)
            
            # Print the number of training parameters in the model
            self.count_params(self.attack_model)
                    
        # Custom CNN
        if self.model_type == "customcnn":
            self.attack_model = nn.Sequential(
                nn.Conv2d(in_channels=1,out_channels=16,kernel_size=2,stride=1,padding=0),
                nn.LeakyReLU(),
                # nn.MaxPool2d(kernel_size=1,stride=1),
                nn.Flatten(),
                nn.Linear(16*2*7,out_features=(2*(m**2) + n**2))
            )
            
            # Print the number of training parameters in the model
            self.count_params(self.attack_model)        
            
    def forward(self, data):
        
        if self.model_type == "linear":
            out = self.attack_model(data.view(data.size(0),-1))
            return out.reshape(self.batch_size,self.linear_output_dim)
        else:
            out = self.attack_model(data)
            return out
        
    def count_params(Self, model):
        # Print the number of training parameters in the model
        num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"This model has {num_param} parameters")