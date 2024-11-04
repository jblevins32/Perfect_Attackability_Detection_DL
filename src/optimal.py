import numpy as np
from scipy.optimize import minimize
from scipy.signal import place_poles

class optimal_attack_matrices:
    def __init__(self, initial_guess, tol = 1e-4):
        '''
        Class for determining attack matrices with conventional numerical optimization technique
        
        Args:
            initial_guess (nparray): Initial state guess by attacker (1xn)
            
        Returns:
            None
        '''
        
        self.initial_guess = initial_guess
        self.tolerance = tol
        
    def objective_func(self,x):
        '''
        Objective function to minimize numerically
        
        Args: 
            x (nparray): input data, firstly set as the initial guess
            
        Returns:
            None
        '''
        
        # Define state space
        A = np.array([[-2,1],
                      [0,1]])
        B = np.array([[1],
                      [1]])
        
        # Define desired poles
        desired_poles = [-3, -4]
        K = place_poles(A,B,desired_poles).gain_matrix
        
        Sx = x[0:4].reshape(A.shape[0], A.shape[1])
        Su = x[4]
        
        transformed = Sx @ (A - (B * Su) @ K @ Sx) @ np.linalg.inv(Sx)
        
        diff = np.linalg.norm(A - (B @ K) - transformed)
        
        return diff
    
    def optimize(self):   
        '''
        Numerical optimization
        
        Args:
            None
            
        Returns:
            tuple: (optimal Sx, optimal Su)
        '''
                     
        # Perform optimal search
        options = {'maxiter': 10000, 'disp': True, 'tol': self.tolerance}
        result = minimize(self.objective_func, initial_guess, method='Nelder-Mead', options=options)
        
        diff = result.fun
        Sx_opt = result.x[0:4].reshape(2,2)
        Su_opt = result.x[-1]
        
        return (Sx_opt, Su_opt), diff
        
        
initial_guess = np.array([-2,1,-2,0,1])
tol = 1e-4
optimizer = optimal_attack_matrices(initial_guess,tol)
result, diff = optimizer.optimize()
