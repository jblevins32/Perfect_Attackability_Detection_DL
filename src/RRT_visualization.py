import numpy as np
from numpy.linalg import eig, inv, cond
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define system matrices
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

# Original closed loop matrix and eigenvalues
Acl = A - B @ K @ C
orig_eig = np.sort(np.linalg.eigvals(Acl))

# Parameters for random tree search
max_branches = 1
branch_scales = [0.5, 1, 2, 3]
max_depth = 1

# Base solution matrices
Sx_base = np.array([[-4.0740, -9.9886],
                    [-0.2515, 0.3862]])
Su_base = np.array([[5.6081, 9.9861],
                    [-3.3484, -6.0484]])
Sxp_base = np.array([[-2.1083, -6.0325, -3.0284],
                     [-0.2453, 0.3850, -0.2534],
                     [-1.9618, -3.9564, -1.0492]])

# Define objective function
def objective(x, A, B, C, K, x0):
    Sx = x[:4].reshape(2, 2)
    Su = x[4:8].reshape(2, 2)
    Sxp = x[8:17].reshape(3, 3)
    
    if abs(np.linalg.det(Sxp)) < 1e-12 or cond(Sxp) > 1e12:
        return 1e12
    
    dx = (np.eye(C.shape[0]) - Sx) @ C @ x0
    dxp = (np.eye(A.shape[0]) - Sxp) @ x0
    du = np.linalg.lstsq(Sxp @ B, Sxp @ (A - B @ Su @ K @ Sx @ C) @ inv(Sxp) @ dxp + Sxp @ B @ Su @ K @ dx, rcond=None)[0]
    
    inv_error = np.linalg.norm(Sxp @ (A - B @ Su @ K @ Sx @ C) @ inv(Sxp) - (A - B @ K @ C), ord='fro')**2
    obs_error = np.linalg.norm(C @ Sxp - Sx @ C, ord='fro')**2
    eig_error = np.linalg.norm(np.sort(eig(A - B @ Su @ K @ Sx @ C)[0]) - np.sort(orig_eig))**2
    drift_error = np.linalg.norm(-Sxp @ (A - B @ Su @ K @ Sx @ C) @ inv(Sxp) @ dxp - Sxp @ B @ Su @ K @ dx + Sxp @ B @ du)**2
    
    identity_penalty = 1.0 / (np.linalg.norm(Sx - np.eye(2), ord='fro')**2 + 1e-8) + \
                       1.0 / (np.linalg.norm(Su - np.eye(2), ord='fro')**2 + 1e-8) + \
                       1.0 / (np.linalg.norm(Sxp - np.eye(3), ord='fro')**2 + 1e-8)
    
    cond_penalty = (cond(Sx) + cond(Su) + cond(Sxp)) / 1e3
    return inv_error + obs_error + eig_error + 2000 * drift_error + 0.15 * identity_penalty + 0.1 * cond_penalty

# Set up the initial solution for tree search
solution_tree = [{
    'Sx': Sx_base,
    'Su': Su_base,
    'Sxp': Sxp_base,
    'depth': 0,
    'branch_scale': 0
}]

# Tree search
for depth in range(max_depth):
    print(f'Exploring depth: {depth}')
    new_solutions = []
    for sol in solution_tree:
        if sol['depth'] < max_depth:
            for scale in branch_scales:
                for _ in range(max_branches):
                    Sx_new = sol['Sx'] + scale * np.random.randn(2, 2)
                    Su_new = sol['Su'] + scale * np.random.randn(2, 2)
                    Sxp_new = sol['Sxp'] + scale * np.random.randn(3, 3)
                    
                    s0 = np.hstack([Sx_new.flatten(), Su_new.flatten(), Sxp_new.flatten()])
                    res = minimize(objective, s0, args=(A, B, C, K, x0), bounds=[(-15, 15)]*len(s0), tol=1e-10)
                    
                    if res.success:
                        Sx = res.x[:4].reshape(2, 2)
                        Su = res.x[4:8].reshape(2, 2)
                        Sxp = res.x[8:17].reshape(3, 3)
                        new_solutions.append({'Sx': Sx, 'Su': Su, 'Sxp': Sxp, 'depth': sol['depth'] + 1, 'branch_scale': scale})
    solution_tree.extend(new_solutions)

# # Plotting the eigenvectors
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# V_A = eig(A)[1]

# colors = ['r', 'g', 'b']
# for i, sol in enumerate(solution_tree):
#     Sxp = sol['Sxp']
#     V_Sxp = eig(Sxp)[1]
#     for j in range(3):
#         origin = V_A[:, j].real
#         direction = V_Sxp[:, j].real
#         end_point = origin + direction
#         ax.plot([origin[0], end_point[0]], [origin[1], end_point[1]], [origin[2], end_point[2]], colors[j] + 'o-')

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.title('3D Plot of Eigenvector Endpoints')
# plt.show()
