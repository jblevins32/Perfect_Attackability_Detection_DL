import numpy as np
import torch.nn as nn
import time
import torch
from torch.utils.data import DataLoader
import pathlib
import yaml
from model import MyModel
from ss_loss import SSLoss

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class DetermineAttackability:
    def __init__(self, **kwargs):
        '''
        Determine Sx,Su,Dx,Du for perfect undetectability
        
        Args:

        Returns:

        '''
        
        # Define some parameters if they are not passed in, and add all to object
        self.path_prefix = kwargs.pop("path_prefix", ".")
        self.imbalance = kwargs.pop("imbalance", "regular")
        self.batch_size = kwargs.pop("batch_size", 128)
        self.model_type = kwargs.pop("model", "TwoLayerNet")
        self.device = kwargs.pop("device", "cpu")
        self.loss_type = kwargs.pop("loss_type", "CE")
        self.lr = kwargs.pop("learning_rate", 0.0001)
        self.momentum = kwargs.pop("momentum", 0.9)
        self.reg = kwargs.pop("reg", 0.0005)
        self.beta = kwargs.pop("beta", 0.9999)
        self.gamma = kwargs.pop("gamma", 1.0)
        self.steps = kwargs.pop("steps", [6, 8])
        self.epochs = kwargs.pop("epochs", 10)
        self.warmup = kwargs.pop("warmup", 0)
        self.save_best = kwargs.pop("save_best", True)
        
        # Define the NN model
        self.model = MyModel()
        print(self.model)

        # Move the model to the given device
        self.model = self.model.to(self.device)

        # Define the optimizer
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            self.lr,
            momentum=self.momentum,
            weight_decay=self.reg,
        )
        
        # Define the criterion / Loss function and send to device
        self.criterion = nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(self.device)
                
        # Reset items
        self.best = 0
        self.best_cm = None # Confusion matrix
        self.best_model = None
        
    def forward(self, data):
        '''
        Forward pass on the data
        
        Args: 
            data to train or test the model on
        
        Returns:
            trained model
            
        '''
        return self.model(data)
    
    def train(self, data):
        for epoch in range(self.epochs):
            iter_time = AverageMeter()
            losses = AverageMeter()
            acc = AverageMeter()
    
            # Put the model into training mode (versus eval mode)
            self.model.train()

            # Train on all data by batch. Note: enumerate() provides both the idx and data
            for idx, data_batch in enumerate(data):
                # Time it
                start = time.time()
                
                # Gather data to put into model
                data_batch = data_batch.to(self.device)
                
                # Get loss and update the model
                out, loss = self._compute_loss_update_params(data_batch, target)
                
                # Check accuracy
                batch_acc = self._check_accuracy(out, target)

                # Losses updating and printing
                losses.update(loss.item(), out.shape[0])
                acc.update(batch_acc, out.shape[0])

                iter_time.update(time.time() - start)
                if idx % 10 == 0:
                    print(
                        (
                            "Epoch: [{0}][{1}/{2}]\t"
                            "Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t"
                            "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                            "Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t"
                        ).format(
                            epoch,
                            idx,
                            len(self.train_loader),
                            iter_time=iter_time,
                            loss=losses,
                            top1=acc,
                        )
                    )
        
        # Save the model
        if self.save_best:
            basedir = pathlib.Path(__file__).parent.resolve()
            torch.save(
                self.best_model.state_dict(),
                str(basedir) + "/checkpoints/" + self.model_type.lower() + ".pth",
            )
                
    def _compute_loss_update_params(self, data, target):
        '''
        Computee the loss, update gradients, and get the output of the model
        
        Args:
            data: input data to model
            target: true labels
            
        Returns:
            output: output of model
            loss: loss value from data
        '''
        
        output = None
        loss = None
        
        # If in training mode, update weights, otherwise do not
        if self.model.training:

            # Main forward pass of the training algorithm
            output = self.model(data)
            
            # Get loss by using output parameters and comparing A-BKC vs attacked model
            A = data[:,:,:1]
            B = data[:,:,2]
            C = 1
            K_transpose = data[:,:,3]
            init_cond = data[:,:,4]
            
            # Calculate loss
            loss = SSLoss(output,A,B,C,K_transpose,init_cond)
            
            # Main backward pass to Update gradients
            self.optimizer.zero_grad()
            loss.backward() # Compute gradients of all the parameters wrt the loss
            self.optimizer.step() # Takes a SGD optimization step

        else:
            
            # Do not update gradients
            with torch.no_grad():
                output = self.model(data)
                loss = nn.CrossEntropyLoss(output, target)

        return output, loss
    
    def _check_accuracy(self, output, target):
        batch_size = target.shape[0]
        
        # Get predicted class from output
        _, pred = torch.max(output, dim=-1)
        
        correct = pred.eq(target).sum() * 1.0
        
        acc = correct / batch_size
        
        return acc