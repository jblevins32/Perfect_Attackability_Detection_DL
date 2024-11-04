# Perfect Attackability Detection for Linear Systems with Deep Learning
This project seeks to find a solution for determining perfect attack matrices for LTI output feedback systems. It utilizes various NN models to accomplish this task.
# File Structure:
- `src`: Source code
  - `attackability_model.py`: Core function for training the model, called from `run.py`
  - `model.py`: NN model definition
  - `optimal.py`: Classic numerical optimization to compare NN model to
  - `run.py`: Generates inputs for the model and runs the training
  - `ss_loss.py`: Loss function
  - `state_space_generator.py`: Data generator for state space models to train/test/evaluate on
- `config.yaml`: Model training parameters

