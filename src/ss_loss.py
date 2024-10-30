import numpy as np

def SSLoss(output,A,B,C,K_transposed,init_cond):
    Sx = output[1]
    Su = output[1]
    Sxp = output[1]
    
    nominal = A - B*K_transposed*C
    attacked = Sxp*(A-B*Su*K_transposed*Sx*C)*inv(Sxp)
    
    error = 1/2*(nominal-attacked)**2
    