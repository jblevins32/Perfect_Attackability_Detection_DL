'''
Evaluate a trained model
'''

import torch
from model import MyModel
from state_space_generator import StateSpaceGenerator
import numpy as np

from compute_loss_update_params import ComputeLossUpdateParams

# Instantiate model
model = MyModel(3,2,1)

# Load model weights in eval mode: choose model path as needed!
model_path = 'models/model_-0.3892.pth'
model.load_state_dict(torch.load(model_path))
model.eval()

# Single input data should be of size (1,1,3,7)
SSgen = StateSpaceGenerator(1) # Generate 1 data piece
data = SSgen.generate(mat_size=3, input_size=2, max_val=10)
data = data.astype(np.float32)
data = torch.from_numpy(data)

# Run inference
with torch.no_grad():
    output = model(data)

# Get loss of eval
output, loss = ComputeLossUpdateParams(data, model, 3, 2)


    
