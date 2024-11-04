import torch
import numpy as np

def PartitionMats(data, output, n, m):
    A = data[:,:,:,0:n].reshape(-1,n,n)
    B = data[:,:,:,n:n+m].reshape(-1,n,m)
    K = data[:,:,:,n+m:n + 2*m].reshape(-1,n,m).transpose(1,2)
    # init_cond = data[:,:,:,n + 2*m:]
    C = np.eye(n) #np.array([[1,0,0],[0,1,0]])
    C = C.astype(np.float32)  # Ensure it's a compatible dtype
    C = torch.from_numpy(C)

    Sxp = output[:,0:n**2].reshape(-1,n,n)
    Su = output[:,n**2:n**2 + m*m].reshape(-1,m,m)
    Sx = output[:,n**2 + m*m:].reshape(-1,n,n)
    
    return Sxp,Su,Sx,A,B,C,K