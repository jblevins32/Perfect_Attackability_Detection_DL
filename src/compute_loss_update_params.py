import torch
from partition_mats import PartitionMats
from ss_loss import SSLoss

def ComputeLossUpdateParams(data, model, n, m, p, optimizer = None):
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
    if model.training:

        # Call the forward pass on the model. The data model() automatically calls model.forward()
        output = model(data)
        
        # Partition out relevent matrices
        Sxp,Su,Sx,A,B,C,K,ic = PartitionMats(data,output,n,m,p)

        # Calculate loss
        loss, diff = SSLoss(Sxp,Su,Sx,A,B,C,K,ic)
        
        # Main backward pass to Update gradients
        optimizer.zero_grad()
        loss.backward() # Compute gradients of all the parameters wrt the loss
        optimizer.step() # Takes a optimization step
        
        return output, loss

    else:
        
        # Do not update gradients
        with torch.no_grad():
            output = model(data)
            
            # Partition out relevent matrices
            Sxp,Su,Sx,A,B,C,K = PartitionMats(data,output,n,m)

            # Calculate loss
            loss, diff = SSLoss(Sxp,Su,Sx,A,B,C,K)
            
            print(f'Sx = \n{Sx}')
            print(f'diff = \n{diff}')

            return output, loss

