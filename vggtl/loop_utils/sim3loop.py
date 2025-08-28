import numpy as np
import torch
import pypose as pp
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple

cpp_version = False
try:
    import sim3solve
    cpp_version = True
except Exception as e:
    print(f"Sim3solve of C++ Version failed, Will using Python Version.")

from fastloop.solve_python import solve_system_py

import time

class Sim3LoopOptimizer:
    """
    Loop closure optimizer for sequences of Sim3 transformations
    
    Input:
    - sequential_transforms: List[Tuple[float, np.ndarray, np.ndarray]]
      Each element is (s, R, t), where s is scalar scale, R is [3,3] rotation matrix, t is [3,] translation vector
    - loop_constraints: List[Tuple[int, int, Tuple[float, np.ndarray, np.ndarray]]]
      Each element is (i, j, (s, R, t)), representing a loop closure constraint from frame i to frame j
    
    Output:
    - Optimized sequential_transforms
    """
    
    def __init__(self, config, device='cpu'):
        self.device = device
        self.config = config
        self.solve_system_version = self.config['Loop']['SIM3_Optimizer']['lang_version'] # choose between 'python' and 'cpp'

        if not cpp_version:
            self.solve_system_version = 'python'
    
    def numpy_to_pypose_sim3(self, s: float, R_mat: np.ndarray, t_vec: np.ndarray) -> pp.Sim3:
        """Convert numpy s,R,t to pypose Sim3"""
        q = R.from_matrix(R_mat).as_quat()  # [x,y,z,w]
        # pypose requires [t, q, s] format
        data = np.concatenate([t_vec, q, np.array([s])])
        return pp.Sim3(torch.from_numpy(data).float().to(self.device))
    
    def pypose_sim3_to_numpy(self, sim3: pp.Sim3) -> Tuple[float, np.ndarray, np.ndarray]:
        """Convert pypose Sim3 to numpy s,R,t"""
        data = sim3.data.cpu().numpy()
        t = data[:3]
        q = data[3:7]  # [x,y,z,w]
        s = data[7]
        R_mat = R.from_quat(q).as_matrix()
        return s, R_mat, t
    
    def sequential_to_absolute_poses(self, sequential_transforms: List[Tuple[float, np.ndarray, np.ndarray]]) -> torch.Tensor:
        """
        Convert sequential relative transforms to absolute pose sequence
        S_01, S_12, S_23, ... -> T_0, T_1, T_2, T_3, ...
        Where T_i is the transform from world coordinate to frame i
        """
        n = len(sequential_transforms) + 1
        poses = []

        identity = pp.Sim3(torch.tensor([0., 0., 0., 0., 0., 0., 1., 1.], device=self.device))
        poses.append(identity)
        
        current_pose = identity
        for s, R_mat, t_vec in sequential_transforms:
            rel_transform = self.numpy_to_pypose_sim3(s, R_mat, t_vec)
            current_pose = current_pose @ rel_transform
            poses.append(current_pose)
        
        return torch.stack(poses)
    
    def absolute_to_sequential_transforms(self, absolute_poses: pp.Sim3) -> List[Tuple[float, np.ndarray, np.ndarray]]:
        """
        Convert absolute pose sequence back to sequential relative transforms
        T_0, T_1, T_2, ... -> S_01, S_12, S_23, ...
        """
        sequential_transforms = []
        n = absolute_poses.shape[0]
        
        for i in range(n - 1):
            rel_transform = absolute_poses[i].Inv() @ absolute_poses[i + 1]
            s, R_mat, t_vec = self.pypose_sim3_to_numpy(rel_transform)
            sequential_transforms.append((s, R_mat, t_vec))
        
        return sequential_transforms
    
    def SE3_to_Sim3(self, x: torch.Tensor) -> pp.Sim3:
        """Convert SE3 to Sim3 (add unit scale)"""
        ones = torch.ones_like(x[..., :1])
        out = torch.cat((x, ones), dim=-1)
        return pp.Sim3(out)
    
    def build_loop_constraints(self, 
                             loop_constraints: List[Tuple[int, int, Tuple[float, np.ndarray, np.ndarray]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build loop closure constraints"""
        if not loop_constraints:
            return torch.empty(0, 8, device=self.device), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        
        loop_transforms = []
        ii_loop = []
        jj_loop = []
        
        for i, j, (s, R_mat, t_vec) in loop_constraints:
            loop_sim3 = self.numpy_to_pypose_sim3(s, R_mat, t_vec)
            loop_transforms.append(loop_sim3.data)
            ii_loop.append(i)
            jj_loop.append(j)
        
        dSloop = pp.Sim3(torch.stack(loop_transforms))
        ii_loop = torch.tensor(ii_loop, dtype=torch.long, device=self.device)
        jj_loop = torch.tensor(jj_loop, dtype=torch.long, device=self.device)
        
        return dSloop, ii_loop, jj_loop
    
    def residual(self, Ginv, input_poses, dSloop, ii, jj, jacobian=False):
        """Compute residuals (modified from original code)"""
        def _residual(C, Gi, Gj):
            out = C @ pp.Exp(Gi) @ pp.Exp(Gj).Inv()
            return out.Log().tensor()
        
        pred_inv_poses = pp.Sim3(input_poses).Inv()
        
        n, _ = pred_inv_poses.shape
        if n > 1:
            kk = torch.arange(1, n, device=self.device)
            ll = kk - 1
            Ti = pred_inv_poses[kk]
            Tj = pred_inv_poses[ll]
            dSij = Tj @ Ti.Inv()
        else:
            kk = torch.empty(0, dtype=torch.long, device=self.device)
            ll = torch.empty(0, dtype=torch.long, device=self.device)
            dSij = pp.Sim3(torch.empty(0, 8, device=self.device))
        
        constants = torch.cat((dSij.data, dSloop.data), dim=0) if dSloop.shape[0] > 0 else dSij.data
        if constants.shape[0] > 0:
            constants = pp.Sim3(constants)
            iii = torch.cat((kk, ii))
            jjj = torch.cat((ll, jj))
            resid = _residual(constants, Ginv[iii], Ginv[jjj])
        else:
            iii = torch.empty(0, dtype=torch.long, device=self.device)
            jjj = torch.empty(0, dtype=torch.long, device=self.device)
            resid = torch.empty(0, device=self.device)
        
        if not jacobian:
            return resid
        
        if constants.shape[0] > 0:
            def batch_jacobian(func, x):
                def _func_sum(*x):
                    return func(*x).sum(dim=0)
                _, b, c = torch.autograd.functional.jacobian(_func_sum, x, vectorize=True)
                from einops import rearrange
                return rearrange(torch.stack((b, c)), 'N O B I -> N B O I', N=2)
            
            J_Ginv_i, J_Ginv_j = batch_jacobian(_residual, (constants, Ginv[iii], Ginv[jjj]))
        else:
            J_Ginv_i = torch.empty(0, device=self.device)
            J_Ginv_j = torch.empty(0, device=self.device)
        
        return resid, (J_Ginv_i, J_Ginv_j, iii, jjj)
    
    def optimize(self, 
                sequential_transforms: List[Tuple[float, np.ndarray, np.ndarray]],
                loop_constraints: List[Tuple[int, int, Tuple[float, np.ndarray, np.ndarray]]],
                max_iterations: int = None,
                lambda_init: float = None) -> List[Tuple[float, np.ndarray, np.ndarray]]:
        """
        Main optimization function
        
        Args:
            sequential_transforms: Input sequence of transforms
            loop_constraints: List of loop closure constraints
            max_iterations: Maximum iterations
            lambda_init: Initial lambda for L-M algorithm
        
        Returns:
            Optimized sequence of transforms
        """
        if max_iterations is None:
            max_iterations = self.config['Loop']['SIM3_Optimizer']['max_iterations']
        if lambda_init is None:
            lambda_init = eval(self.config['Loop']['SIM3_Optimizer']['lambda_init'])

        input_poses = self.sequential_to_absolute_poses(sequential_transforms)
        
        dSloop, ii_loop, jj_loop = self.build_loop_constraints(loop_constraints)
        
        if len(loop_constraints) == 0:
            print("Warning: No loop constraints provided, returning original transforms")
            return sequential_transforms
        
        Ginv = pp.Sim3(input_poses).Inv().Log()
        lmbda = lambda_init
        residual_history = []
        
        print(f"Starting optimization with {len(sequential_transforms)} poses and {len(loop_constraints)} loop constraints")
        
        # L-M loop
        for itr in range(max_iterations):
            resid, (J_Ginv_i, J_Ginv_j, iii, jjj) = self.residual(
                Ginv, input_poses, dSloop, ii_loop, jj_loop, jacobian=True)
            
            if resid.numel() == 0:
                print("No residuals to optimize")
                break
                
            current_cost = resid.square().mean().item()
            residual_history.append(current_cost)
            
            try: # Solve linear system
                begin_time = time.time()
                if self.solve_system_version == 'cpp':
                    delta_pose, = sim3solve.solve_system(
                        J_Ginv_i, J_Ginv_j, iii, jjj, resid, 0.0, lmbda, -1)
                elif self.solve_system_version == 'python':
                    delta_pose = solve_system_py(
                        J_Ginv_i, J_Ginv_j, iii, jjj, resid, 0.0, lmbda, -1)
                else:
                    print(f"Solver version has not been chosen! ('python' or 'cpp')")
                end_time = time.time()
            except Exception as e:
                print(f"Solver failed at iteration {itr}: {e}")
                break
            
            Ginv_tmp = Ginv + delta_pose
            
            new_resid = self.residual(Ginv_tmp, input_poses, dSloop, ii_loop, jj_loop)
            new_cost = new_resid.square().mean().item() if new_resid.numel() > 0 else float('inf')
            
            # L-M
            if new_cost < current_cost:
                Ginv = Ginv_tmp
                lmbda /= 2
                print(f"Iteration {itr}: cost {current_cost:.14f} -> {new_cost:.14f} (accepted)", end=' | ')
            else:
                lmbda *= 2
                print(f"Iteration {itr}: cost {current_cost:.14f} -> {new_cost:.14f} (rej)     ", end=' | ') # more readible to accepted
            
            print(f'Time of solver ({self.solve_system_version}): {(end_time - begin_time)*1000:.4f} ms')

            if (current_cost < 1e-5) and (itr >= 4):
                if len(residual_history) >= 5:
                    improvement_ratio = residual_history[-5] / residual_history[-1]
                    if improvement_ratio < 1.5:
                        print(f"Converged at iteration {itr}")
                        break
        
        optimized_absolute_poses = pp.Exp(Ginv).Inv()
        
        optimized_sequential = self.absolute_to_sequential_transforms(optimized_absolute_poses)
        
        print(f"Optimization completed. Final cost: {residual_history[-1] if residual_history else 'N/A'}")
        
        return optimized_sequential


# ======== TEST CODE ========

from scipy.spatial.transform import Rotation as R

def create_ring_transforms(num_poses=6, radius=5.0, rot_noise_deg=2.0):
    """Generate a ring of Sim3 transforms with rotation, adding slight rotational noise"""
    transforms = []
    angle_step = 2 * np.pi / num_poses

    for i in range(num_poses):
        angle = angle_step

        # Main rotation (around Z-axis)
        R_z = R.from_euler('z', angle, degrees=False)

        # Add slight rotational noise (Gaussian noise in degrees)
        noise_angles_deg = np.random.normal(loc=0.0, scale=rot_noise_deg, size=3)
        R_noise = R.from_euler('xyz', noise_angles_deg, degrees=True)

        # Combine rotations
        R_mat = (R_noise * R_z).as_matrix()

        # Translation: simulate a circular trajectory
        t = np.array([radius * np.sin(angle), radius * (1 - np.cos(angle)), 0.0])

        s = np.random.uniform(0.8, 1.2)

        transforms.append((s, R_mat, t))

    return transforms

def example_usage():
    optimizer = Sim3LoopOptimizer(solve_system_version='cpp')

    # Build rotating ring
    sequential_transforms = create_ring_transforms(num_poses=20, radius=3.0)
    
    # Add loop closure constraint: from frame 5 back to frame 0
    loop_constraints = [
        (20, 0, (1.0, np.eye(3), np.zeros(3)))  # Temporary unit loop for simulation
    ]
    
    # Trajectory before/after optimization
    input_abs_poses = optimizer.sequential_to_absolute_poses(sequential_transforms)
    optimized_transforms = optimizer.optimize(sequential_transforms, loop_constraints)
    optimized_abs_poses = optimizer.sequential_to_absolute_poses(optimized_transforms)

    def extract_xyz(pose_tensor):
        poses = pose_tensor.cpu().numpy()
        return poses[:, 0], poses[:, 1], poses[:, 2]
    
    x0, y0, z0 = extract_xyz(input_abs_poses)
    x1, y1, z1 = extract_xyz(optimized_abs_poses)

    # Visualize trajectory
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    plt.figure(figsize=(8, 6))
    plt.plot(x0, y0, 'o--', label='Before Optimization')
    plt.plot(x1, y1, 'o-', label='After Optimization')
    for i, j, _ in loop_constraints:
        plt.plot([x0[i], x0[j]], [y0[i], y0[j]], 'r--', label='Loop (Before)' if i == 5 else "")
        plt.plot([x1[i], x1[j]], [y1[i], y1[j]], 'g-', label='Loop (After)' if i == 5 else "")
    plt.gca().set_aspect('equal')
    plt.title("Sim3 Loop Closure Optimization (Rotating Ring)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

    return optimized_transforms

if __name__ == "__main__":
    example_usage()