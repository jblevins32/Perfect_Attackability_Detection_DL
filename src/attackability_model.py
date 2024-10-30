import numpy as np
import torch.nn as nn
import time
import torch
import pathlib
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
        self.model = MyModel(self.n,self.m)
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
                
        # Reset items
        # self.best = 0
        # self.best_cm = None # Confusion matrix
        # self.best_model = None
    
    def train(self, data):
        for epoch in range(self.epochs):
            # Adjust learning rate
            self._adjust_learning_rate(epoch)
            
            # Initialize a meter for printing info
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
                out, loss = self._compute_loss_update_params(data_batch)
                
                # Check accuracy
                # batch_acc = self._check_accuracy(out)

                # Losses updating and printing
                losses.update(loss.item(), out.shape[0])
                # acc.update(batch_acc, out.shape[0])

                # Update time and info every 10 epochs
                iter_time.update(time.time() - start)
                if idx % 10 == 0:
                    print(
                        (
                            "Epoch: [{0}][{1}]\t"
                            "Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t"
                            "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                            # "Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t"
                        ).format(
                            epoch,
                            idx,
                            # len(self.train_loader),
                            iter_time=iter_time,
                            loss=losses,
                            # top1=acc,
                        )
                    )
        
        # Save the model
        if self.save_best:
            basedir = pathlib.Path(__file__).parent.resolve()
            torch.save(
                self.best_model.state_dict(),
                str(basedir) + "/checkpoints/" + self.model_type.lower() + ".pth",
            )
                
    def _compute_loss_update_params(self, data):
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

            # Call the forward pass on the model. The data model() automatically calls model.forward()
            output = self.model(data)
            
            # Partition out relevent matrices
            A = data[:,:,:,0:self.n].reshape(-1,self.n,self.n)
            B = data[:,:,:,self.n:self.n+self.m].reshape(-1,self.n,self.m)
            K_transpose = data[:,:,:,self.n+self.m:self.n + 2*self.m].reshape(-1,self.m,self.n) 
            init_cond = data[:,:,:,self.n + 2*self.m:]
            C = np.eye(self.n) #np.array([[1,0,0],[0,1,0]])
            C = C.astype(np.float32)  # Ensure it's a compatible dtype
            C = torch.from_numpy(C)
            
            Sxp = output[:,0:self.n**2].reshape(-1,self.n,self.n)
            Su = output[:,self.n**2:self.n**2 + self.m*self.m].reshape(-1,self.m,self.m)
            Sx = output[:,self.n**2 + self.m*self.m:].reshape(-1,self.n,self.n)

            # Calculate loss
            loss = SSLoss(Sxp,Su,Sx,A,B,C,K_transpose,init_cond)
            
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
            
    # def _check_accuracy(self, output, target):
    #     batch_size = target.shape[0]
        
    #     # Get predicted class from output
    #     _, pred = torch.max(output, dim=-1)
        
    #     correct = pred.eq(target).sum() * 1.0
        
    #     acc = correct / batch_size
        
    #     return acc