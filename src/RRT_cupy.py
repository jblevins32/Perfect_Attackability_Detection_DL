import numpy as np
import cupy as cp  # GPU-accelerated equivalent of NumPy
from scipy.optimize import minimize
import pandas as pd
from joblib import Parallel, delayed

# System matrices on CPU (e.g., CuPy doesn't support eigvals directly)
A = np.array([[-2, 1, 0],
              [0, -1, 1],
              [1, 0, -3]])
B = np.array([[1, 0],
              [0, 1],
              [1, 1]])
C = np.array([[1, 0, 1],
              [0, 1, 0]])
K = np.array([[-0.5, 2],
              [0.3, -0.8]])
x0 = np.array([[1], [0], [0]])

# Original closed-loop eigenvalues
Acl = A - B @ K @ C
orig_eig = np.sort(np.linalg.eigvals(Acl))

# Base solution matrices
Sx_base = cp.array([[-4.0740, -9.9886],
                    [-0.2515, 0.3862]])
Su_base = cp.array([[5.6081, 9.9861],
                    [-3.3484, -6.0484]])
Sxp_base = cp.array([[-2.1083, -6.0325, -3.0284],
                     [-0.2453, 0.3850, -0.2534],
                     [-1.9618, -3.9564, -1.0492]])

# Objective function adapted for GPU with CuPy
def objective(x):
    # ... Your objective function remains the same, but replace np with cp ...
    # Use CuPy where possible for matrix operations
    # Example:
    Sx = cp.reshape(x[:4], (2, 2))
    Su = cp.reshape(x[4:8], (2, 2))
    Sxp = cp.reshape(x[8:17], (3, 3))
    
    # Objective function calculations (ensure all are converted for CuPy compatibility)
    # Return objective value

# Tree search with parallel processing
def search_solution(Sx, Su, Sxp, scale):
    Sx_new = Sx + scale * cp.random.randn(2, 2)
    Su_new = Su + scale * cp.random.randn(2, 2)
    Sxp_new = Sxp + scale * cp.random.randn(3, 3)
    s0 = cp.hstack([Sx_new.flatten(), Su_new.flatten(), Sxp_new.flatten()])
    
    # Run optimization
    res = minimize(objective, s0.get(), bounds=[(-15, 15)]*len(s0), tol=1e-10)  # Use .get() to convert to NumPy for SciPy
    
    if res.success:
        return {
            'Sx': res.x[:4].reshape(2, 2),
            'Su': res.x[4:8].reshape(2, 2),
            'Sxp': res.x[8:17].reshape(3, 3),
            'scale': scale
        }

# Generate solutions in parallel
branch_scales = [0.5, 1, 2, 3]
solutions = Parallel(n_jobs=-1)(
    delayed(search_solution)(Sx_base, Su_base, Sxp_base, scale)
    for scale in branch_scales
)

# Create a DataFrame and save
df = pd.DataFrame([
    {'scale': sol['scale'], 'Sx': sol['Sx'].flatten().tolist(),
     'Su': sol['Su'].flatten().tolist(), 'Sxp': sol['Sxp'].flatten().tolist()}
    for sol in solutions if sol is not None
])
df.to_excel('data/fdia_tree_solutions.xlsx', index=False)

print("Solutions saved to 'data/fdia_tree_solutions.xlsx'")
