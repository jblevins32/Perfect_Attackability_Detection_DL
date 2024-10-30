import torch.nn as nn
import torch
import numpy as np

def SSLoss(Sxp,Su,Sx,A,B,C,K_transposed,init_cond):
    nominal = A - B @ K_transposed @ C
    attacked = Sxp @ (A - B @ Su @ K_transposed @ Sx @ C) @ torch.inverse(Sxp)
    
    eig_nominal = torch.linalg.eig(nominal).eigenvalues.abs()
    eig_attacked = torch.linalg.eig(attacked).eigenvalues.abs()
    
    error = torch.norm(eig_attacked-eig_nominal,p=2)
    
    return error

    # drift = -Sxp*(A-B*Su*K_transposed*Sx*C)*torch.inv(Sxp)*dxp - Sxp*B*Su*K_transposed*dx + Sxp*B*du
    