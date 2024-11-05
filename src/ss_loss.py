import torch.nn as nn
import torch
import numpy as np

def SSLoss(Sxp,Su,Sx,A,B,C,K,ic):
    
    # Removing K placeholder 1s
    K = K[:,:,:-1]
    
    nominal = A - B @ K @ C
    attacked = A - B @ Su @ K @ Sx @ C
    diff = nominal-attacked
    
    attacked_mod = Sxp @ attacked @ torch.inverse(Sxp)
    
    inv = attacked_mod - nominal
    
    # Need to make Sx a 2x2 (change C to not identity matrix)
    Dx = (torch.eye(C.shape[1]) - Sx) @ C @ ic
    Dxp = (torch.eye(A.shape[1]) - Sxp) @ ic
    
    coeff_du = Sxp @ B
    rhs = attacked_mod @ Dxp + Sxp @ B @ Su @ K @ Dx
    Du = torch.linalg.pinv(coeff_du) @ rhs
    
    # Error terms
    inv_error = torch.norm(inv)**2
    # obs_error = torch.norm(C @ Sxp - Sx @ C)**2
    eig_error = torch.norm(torch.sort(torch.linalg.eig(attacked)[0].abs())[0] - torch.sort(torch.linalg.eig(nominal)[0].abs())[0])**2
    drift_error = torch.norm(-rhs + Sxp @ B @ Du)**2
    
    # Identity Penalty
    identity_penalty = 10*(1/(torch.norm(Sx - torch.eye(2))**2 + 1e-6) + 1/(torch.norm(Su - torch.eye(2))**2 + 1e-6) + 1/(torch.norm(Sxp - torch.eye(3))**2 + 1e-6))
        
    # Cross-term penalty
    cross_penalty = 1/(torch.norm(Sx - Su)**2 + 1e-6)
      
    loss = torch.log(inv_error + eig_error + drift_error + identity_penalty + cross_penalty)
     
    # loss = torch.log(torch.norm(diff,p=2)+1e-6)
    # Penalize being close to trivial solution - comparing eigenvalues to diagonal entires of the matrix to 2 decimals
    # if (np.round(np.linalg.eig(Sx.detach().numpy()[0])[0],2) == np.round(np.diag(Sx.detach().numpy()[0]),2)).all():
    #     loss += 1
    
        
    return loss, diff

    
    # drift = -Sxp*(A-B*Su*K_transposed*Sx*C)*torch.inv(Sxp)*dxp - Sxp*B*Su*K_transposed*dx + Sxp*B*du
    