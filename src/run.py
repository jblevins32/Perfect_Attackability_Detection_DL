import yaml
from attackability_model import DetermineAttackability
from state_space_generator import StateSpaceGenerator
from torch.utils.data import DataLoader
import torch
import numpy as np

# Choose calculation device - Use cpu if CUDA gpu not available
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device = " + device)

# Unpacking parameters from config yaml file to kwargs dictionary. Kwargs allows for a function to accept any number of arguments
with open("config.yaml", "r") as read_file:
  config = yaml.safe_load(read_file)

kwargs = {}
for key in config:
    for k, v in config[key].items():
        if k!= 'description':
            kwargs[k] = v

kwargs['device'] = device
print(kwargs)

# Instantiate training object which loads all model parameters
solver = DetermineAttackability(**kwargs)

# Generate state space data... format: [A,B,K.T,initial conditions]
data_gen = StateSpaceGenerator(num_mats=kwargs['num_mats'])
data = data_gen.generate(mat_size=kwargs['system_order'], input_size = kwargs['input_size'],max_val=kwargs['max_val'])

data = data.astype(np.float32)  # Ensure it's a compatible dtype
data = torch.from_numpy(data)

train_loader = DataLoader(
    data, batch_size=kwargs['batch_size'], shuffle=True
)

# Train model
solver.train(train_loader)
