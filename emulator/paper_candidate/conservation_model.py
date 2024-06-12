import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as fd
import numpy as np
import time
from firedrake.petsc import PETSc

class ConservationModel:
    def __init__(
        self, 
        data_mapper, 
        vel_scale=1.,
        len_scale=1e3,
        time_scale=1.,
        thklim=1., 
        theta=1.0, 
        flux_type='upwind',
        solver_type='direct'
    ):
                
        self.data_mapper = data_mapper
        self.mesh = data_mapper.mesh
        nhat = fd.FacetNormal(self.mesh)

        self.V_rt = V_rt = fd.Function(mesh, 'RT', 1)
        self.V_dg = V_dg = fd.Function(mesh, 'DG', 0)

        zeta = self.zeta = fd.Constant(time_scale*vel_scale/len_scale)
        theta = self.theta = fd.Constant(theta)
        self.thklim = thklim

        Ubar = self.Ubar = fd.Function(V_rt, name='Ubar')
        H0 = self.H0 = fd.Function(V_dg, name='H0')
        adot = self.adot = fd.Function(V_dg, name='adot') 
        H = self.H = fd.Function(V_dg, name='H')
        Hmid = theta*H + (1-theta)*H0
        xsi = fd.TestFunction(V_dg)
        dH = fd.TrialFunction(V_dg)

        H_avg = 0.5*(Hmid('+') + Hmid('-'))
        H_jump = Hmid('+')*nhat('+') + Hmid('-')*nhat('-')
        xsi_jump = xsi('+')*nhat('+') + xsi('-')*nhat('-')
        unorm = fd.dot(Ubar, Ubar)**0.5
        dt = self.dt = fd.Constant(1.0)


        # Lax-Friederichs flux
        if flux_type=='centered':
            uH = fd.avg(Ubar)*H_avg

        elif flux_type=='lax-friedrichs':
            uH = fd.avg(Ubar)*H_avg + fd.Constant(0.5)*fd.avg(unorm)*H_jump

        elif flux_type=='upwind':
            uH = fd.avg(Ubar)*H_avg + 0.5*abs(fd.dot(fd.avg(Ubar),nhat('+')))*H_jump

        else:
            print('Invalid flux')

        R_transport = ((H - H0)/dt - adot)*xsi*fd.dx + zeta*fd.dot(uH, xsi_jump)*fd.dS 
        R_lin = self.R_lin = fd.replace(R_transport,{H:dH})
        self.problem = fd.LinearVariationalProblem(fd.lhs(R_lin), fd.rhs(R_lin), H)

        if solver_type=='direct':
            solver_params = {"ksp_type": "preonly",
                                  "pmat_type":"aij",
                                  "pc_type": "lu",  
                                  "pc_factor_mat_solver_type": "mumps"} 
        else:
            solver_params = {'pc_type': 'bjacobi',
                                  "ksp_rtol":1e-5}
            
        self.solver = fd.LinearVariationalSolver(
            self.problem,
            solver_parameters= solver_params
            )

    def step(self, dt, Ubar, H0, adot):
        self.dt.assign(dt)
        self.adot.assign(adot)
        self.Ubar.assign(Ubar)
        self.H0.assign(H0)
        self.solver.solve()

        self.H.dat.data[self.H.dat.data < self.thklim] = self.thklim
        return self.H

         
