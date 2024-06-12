import os
import sys
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.append('./')
import firedrake as df
from firedrake.petsc import PETSc
from speceis_dg.hybrid_new import CoupledModel
import numpy as np
from scipy.special import expit



with df.CheckpointFile(f'mt_training/data/Beaverhead_0_1000/output_Beaverhead_0_1000.h5', 'r') as afile:
    mesh = afile.load_mesh()

    j = 50
    d = {}
    d['B'] = afile.load_function(mesh, 'B')
    d['H'] = afile.load_function(mesh, 'H0', idx=j)
    d['beta2'] = afile.load_function(mesh, 'beta2', idx=j)
    d['Ubar'] = afile.load_function(mesh, 'Ubar0', idx=j)
    d['Udef'] = afile.load_function(mesh, 'Udef0', idx=j)

    vel_scale = 1.
    thk_scale = 1.
    len_scale = 1e3
    beta_scale = 1e4
    
    config = {
            'solver_type': 'direct',
            'sliding_law': 'linear',
            'vel_scale': vel_scale,
            'thk_scale': thk_scale,
            'len_scale': len_scale,
            'beta_scale': beta_scale,
            'theta': 1.0,
            'thklim': 2.,
            'alpha': 1000.0,
            'z_sea': -1000.,
            'calve': False,
            'velocity_function_space' : 'MTW'
        }
          
    model = CoupledModel(mesh,**config)


    model.B.assign(d['B'])
    model.H0.assign(d['H'])
    model.beta2.assign(d['beta2'])
    model.B_grad_solver.solve()
    model.S_grad_solver.solve()

    
    df.solve(model.R_stress_vel == 0,  model.W)


    model.solve_vel()
    #model.step(0., 0.1)
