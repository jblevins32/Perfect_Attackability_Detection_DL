import torch
import numpy as np

def PartitionMats(data, output, n, m, p):
    A = data[:,:,:,0:n].reshape(-1,n,n)
    B = data[:,:,:,n:n+m].reshape(-1,n,m)
    K = data[:,:,:,n+m:n + 2*m].reshape(-1,n,m).transpose(1,2)
    # init_cond = data[:,:,:,n + 2*m:]
    C = data[:,:,:,n + 2*m:n + 2*m + p].reshape(-1,n,p).transpose(1,2)
    ic = data[:,:,:,-1].reshape(-1,n,1)

    Sxp = output[:,0:n**2].reshape(-1,n,n)
    Su = output[:,n**2:n**2 + m*m].reshape(-1,m,m)
    Sx = output[:,n**2 + m*m:].reshape(-1,m,m)
    
    return Sxp,Su,Sx,A,B,C,K,ic