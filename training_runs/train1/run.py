import os
import sys
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.append('../../')
import firedrake as df
import pickle
from firedrake.petsc import PETSc
from speceis_dg.hybrid import CoupledModel
import numpy as np


class Run:
    def __init__(self, i, res=1500):

        results_dir = f'data/{i}_{res}'

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
      
        with df.CheckpointFile(f'data/input_{i}_{res}.h5', 'r') as afile:
            mesh = afile.load_mesh()
            H = afile.load_function(mesh, 'H')
            B = afile.load_function(mesh, 'B')
            adot = afile.load_function(mesh, 'adot')
            beta2 = afile.load_function(mesh, 'beta2')
            c = mesh.coordinates.vector().array()

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
            'thklim': 1.,
            'alpha': 1000.0,
            'z_sea': -1000.,
            'calve': False,
            'velocity_function_space' : 'MTW'
        }
          
        beta2.interpolate(beta2 + df.Constant(0.05))
        model = self.model = CoupledModel(mesh,**config)
        model.beta2.interpolate(beta2)
        model.B.interpolate(B)
        model.H0.interpolate(H + df.Constant(1. / thk_scale))        
        model.adot.interpolate(adot)

        S_file = df.File(f'{results_dir}/S.pvd')
        H_file = df.File(f'{results_dir}/H.pvd')
        beta2_file = df.File(f'{results_dir}/beta2.pvd')
        Ubar_file = df.File(f'{results_dir}/Ubar.pvd')
        Udef_file = df.File(f'{results_dir}/Udef.pvd')
        adot_file = df.File(f'{results_dir}/adot.pvd')

        S_out = df.Function(model.Q_thk,name='S')

        t = 0.
        t_end = 400
        dt = 3.
        max_step = 2.5

        time_step_factor = 1.01

        with df.CheckpointFile(f"{results_dir}/output_{i}.h5", 'w') as afile:
            
            afile.save_mesh(mesh)
            afile.save_function(model.B)

            j = 0
            while t < t_end:
                dt = min(dt*time_step_factor,max_step)

                beta_scale = 1. + (1./3.)*np.cos(t*2.*np.pi / 100.)
                adot0 = 2.*np.sin(t*2.*np.pi / 400. ) 
                model.beta2.interpolate(beta2*df.Constant(beta_scale))
                eps = 1.5*np.sqrt(model.H0.dat.data)*np.random.randn(len(model.adot.dat.data))
                model.adot.interpolate(adot + df.Constant(adot0))
                model.adot.dat.data[:] += eps
                
                converged = model.step(t,
                                       dt,
                                       picard_tol=1e-3,
                                       momentum=0.5,
                                       max_iter=25,
                                       convergence_norm='l2')

                if not converged:
                    dt*=0.5
                    continue
                
                t += dt


                PETSc.Sys.Print('step', t,dt,df.assemble(model.H0*df.dx))
               
                n_out = 1
                if j % n_out == 0:
                    out_idx = int(j/n_out)

                    S_out.assign(df.project(model.S, model.Q_thk))

                    afile.save_function(model.H0, idx=out_idx)
                    afile.save_function(S_out, idx=out_idx)
                    afile.save_function(model.Ubar0, idx=out_idx)
                    afile.save_function(model.Udef0, idx=out_idx)
                    afile.save_function(model.beta2, idx=out_idx)
                    afile.save_function(model.adot, idx=out_idx)

                    S_file.write(S_out, time=t)
                    H_file.write(model.H0,time=t)
                    Ubar_file.write(model.Ubar0, time=t)
                    Udef_file.write(model.Udef0, time=t)
                    adot_file.write(model.adot, time=t)
                    beta2_file.write(model.beta2, time=t)
                
                j += 1
        

if __name__=='__main__':
    i = int(sys.argv[1])
    print(i)
    bc = Run(i)