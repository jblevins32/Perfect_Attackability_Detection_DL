import torch.nn as nn
import torch
import numpy as np

def SSLoss(Sxp,Su,Sx,A,B,C,K):
    nominal = A - B @ K @ C
    attacked = Sxp @ (A - B @ Su @ K @ Sx @ C) @ torch.inverse(Sxp)
    
    # eig_nominal = torch.linalg.eig(nominal).eigenvalues.abs()
    # eig_attacked = torch.linalg.eig(attacked).eigenvalues.abs()
    
    # diff = torch.log(torch.norm(eig_attacked-eig_nominal,p=2))
    
    diff= torch.log(torch.norm(nominal-attacked,p=2)+1e-6)
    return diff

    # drift = -Sxp*(A-B*Su*K_transposed*Sx*C)*torch.inv(Sxp)*dxp - Sxp*B*Su*K_transposed*dx + Sxp*B*du
    