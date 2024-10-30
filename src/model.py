import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.flatten = nn.Flatten()
        # self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size,shuffle=True)

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=1,kernel_size=1,stride=1,padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=1,stride=1),
            nn.BatchNorm2d(1),
            nn.Linear(1024,out_features=10)
        )