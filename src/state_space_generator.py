import numpy as np
from scipy.linalg import solve_continuous_are

'''
This program generates state spaces as data for ML models to train/test on
'''

class StateSpaceGenerator:
    def __init__(self, num_mats):
        '''
        Args:
            num_mats = number of matrices to add to the data
        '''
        
        self.num_mats = num_mats
        
    def generate(self, mat_size, input_size, output_size, max_val):
        '''
        Generates random state space data
        
        Args: 
            mat_size: size of generated matrices
            max_val: largest possible value in the matrices
        
        Returns: 
            data (array): numpy list of arrays of concatenated state space vectors A,B,K, and Initial Conditions (num_data,1,n,n+2m)
        
        '''
        
        # Initialize data vector
        data = []
        mat_counter = self.num_mats
        
        while mat_counter != 0:
            
            # Randomly generate the A matrix, B matrix, C matrix, and initial conditions (ic)
            A = (np.random.rand(mat_size,mat_size)-0.5)*max_val # Generates values between [-max_val/2 , max_val/2]
            B = np.random.rand(mat_size, input_size)*max_val # Generates values between [0 , max_val]
            C = np.random.rand(output_size, mat_size)
            C = (C > 0.5).astype(int) # turn c into binary values only
            ic = np.random.rand(mat_size,1)*max_val 
            
            # Determine stable K with LQR
            # Q = np.eye(output_size)
            # R = np.eye(input_size)
            # P = solve_continuous_are(A,B,Q,R)
            # K = np.linalg.inv(R) @ B.T @ P
            
            K = np.random.rand(input_size, output_size)
            
            # Randomly assign initial conditions
            # initcond: init_cond = np.random.rand(mat_size, 1)*max_val
            
            # Check if the closed loop system is not stable, if so, skip
            if self.check_stability(A, B, K, C):
                continue
        
            # Check if the system is not controllable, if so, skip
            if self.check_controllability(A,B):
                continue
            
            mat_counter -= 1

            cat = np.array([1,1]).reshape(-1,1)            
            K = np.concatenate(((K,cat)),axis=1)
            data.append(np.concatenate((A,B,K.T,C.T,ic),axis=1))

        data = np.array(data, dtype = object)
        
        # Reshape to (total samples, 1 channel, rows, cols)
        # initcond: data = data.reshape(self.num_mats, 1,A.shape[0],A.shape[1]+B.shape[1]+K.shape[0]+init_cond.shape[1])
        data = data.reshape(self.num_mats, 1,A.shape[0],A.shape[1]+B.shape[1]+K.shape[0]+C.shape[0]+ic.shape[1])
            
        return data
    
    def check_stability(self, A, B, K, C):
        stable_bool = (((np.linalg.eig(A-B @ K @ C))[0]).real >= 0).any()
        # if stable_bool: print("A SS was found to be unstable")
        return stable_bool
    
    def check_controllability(self, A, B):
        controllability = []
        for dim in range(A.shape[0]):
            controllability.append(np.linalg.matrix_power(A, dim) @ B)
        
        controllability = np.hstack(controllability) # Combine list into numpy array
        controllability_bool = np.linalg.matrix_rank(controllability) != A.shape[0]
        if controllability_bool: print("A SS was found to be uncontrollable")
        return controllability_bool
            
if __name__ == "__main__":
    ssg = StateSpaceGenerator(num_mats=5)
    data = ssg.generate(mat_size_min=2, mat_size_max=10, max_val=10)
    test=1
    
        
