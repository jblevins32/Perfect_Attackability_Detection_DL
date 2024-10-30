import torch.nn as nn
import torch

def SSLoss(Sxp,Su,Sx,A,B,C,K_transposed,init_cond):
    dx = 1
    dxp = 1
    du = 1
    
    nominal = A - B @ K_transposed @ C
    attacked = Sxp @ (A - B @ Su @ K_transposed @ Sx @ C) @ torch.inverse(Sxp)
    
    mse_loss = nn.MSELoss()
    error = mse_loss(nominal, attacked)
    
    return error

    # drift = -Sxp*(A-B*Su*K_transposed*Sx*C)*torch.inv(Sxp)*dxp - Sxp*B*Su*K_transposed*dx + Sxp*B*du
    