import numpy as np
import torch.nn as nn
import time
import torch
from model import MyModel
import matplotlib.pyplot as plt
from compute_loss_update_params import ComputeLossUpdateParams

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
        Determine Sxp,Sx,Su,Dx,Du for perfect undetectability
        
        Args:
            *kwargs
            
        Returns:
            None
        '''
        
        # Define some parameters if they are not passed in, and add all to object
        self.batch_size = kwargs.pop("batch_size", 10)
        self.device = kwargs.pop("device", "cpu")
        self.lr = kwargs.pop("learning_rate", 0.0001)
        self.momentum = kwargs.pop("momentum", 0.9)
        self.reg = kwargs.pop("reg", 0.0005)
        self.beta = kwargs.pop("beta", 0.9999)
        self.gamma = kwargs.pop("gamma", 1.0)
        self.steps = kwargs.pop("steps", [6, 8])
        self.epochs = kwargs.pop("epochs", 10)
        self.warmup = kwargs.pop("warmup", 0)
        self.save_best = kwargs.pop("save_best", True)
        self.n = kwargs.pop("n", 2)
        self.m = kwargs.pop("m", 1)
        
        # Define the NN model
        self.model = MyModel(self.n,self.m,self.batch_size)
        print(self.model)

        # Move the model to the given device
        self.model = self.model.to(self.device)

        # Define the optimizer
        # self.optimizer = torch.optim.SGD(
        #     self.model.parameters(),
        #     self.lr,
        #     momentum=self.momentum,
        #     weight_decay=self.reg,
        # )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            self.lr,
            weight_decay=self.reg
        )
               
        self.train_losses = []
    
    def train(self, data):
        '''
        Train the model
        
        Args:
            data (nparray): input state space data (num_samples, 1, n, n + 2*m)
            
        Returns:
            None, saves models and figs
        '''
        
        # Main training loop
        for epoch in range(self.epochs):
            
            # Adjust learning rate for SGD
            if isinstance(self.optimizer, torch.optim.SGD):
                self._adjust_learning_rate(epoch)
            
            # Initialize a meter for printing info
            iter_time = AverageMeter()
            losses = AverageMeter()
    
            # Put the model into training mode (versus eval mode)
            self.model.train()

            # Train on all data by batch. Note: enumerate() provides both the idx and data
            for idx, data_batch in enumerate(data):
                
                start = time.time()
                
                # Gather data to be trained on the chosen device
                data_batch = data_batch.to(self.device)
                
                # Get loss and update the model
                out, loss = ComputeLossUpdateParams(data_batch, self.model, self.n, self.m, self.optimizer)
                
                # Losses updating and printing
                losses.update(loss.item(), out.shape[0])

                # Update time and info every 10 epochs
                iter_time.update(time.time() - start)
                if idx % 10 == 0:
                    print(
                        (
                            "Epoch: [{0}][{1}]\t"
                            "Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t"
                            "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        ).format(
                            epoch,
                            idx,
                            iter_time=iter_time,
                            loss=losses,
                        )
                    )
            self.plot(loss)
        
        # Save the model and figs
        model_name = f'models/model_{loss:.4f}.pth'
        fig_name_png = f'figs/loss_{loss:.4f}.png'
        fig_name_eps = f'figs/loss_{loss:.4f}.eps'
        plt.savefig(fig_name_png)
        plt.savefig(fig_name_eps)
        torch.save(self.model.state_dict(), model_name)
        
        # if self.save_best:
        #     basedir = pathlib.Path(__file__).parent.resolve()
        #     torch.save(
        #         self.best_model.state_dict(),
        #         str(basedir) + "/checkpoints/" + self.model_type.lower() + ".pth",
        #     )
        
    def plot(self, loss):
        '''
        Plot loss live during training
        
        Args:
            loss (int): loss at end of each epoch
            
        Returns:
            None, plots loss over epoch
        '''
        
        self.train_losses.append(float(loss.detach().numpy()))
        plt.plot(np.arange(1, len(self.train_losses) + 1), self.train_losses, label ='', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.pause(0.0000001)
    
    def _adjust_learning_rate(self, epoch):
        epoch += 1
        if epoch <= self.warmup:
            lr = self.lr * epoch / self.warmup
        elif epoch > self.steps[1]:
            lr = self.lr * 0.01
        elif epoch > self.steps[0]:
            lr = self.lr * 0.1
        else:
            lr = self.lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr