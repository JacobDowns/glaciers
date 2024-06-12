import os
import sys
#os.environ['OMP_NUM_THREADS'] = '1'
#sys.path.append('../../')
import firedrake as fd
#import pickle
#from firedrake.petsc import PETSc
#from speceis_dg.hybrid import CoupledModel
import numpy as np



with fd.CheckpointFile(f'mt_training/data/input_Beaverhead_1_500.h5', 'r') as afile:
    mesh = afile.load_mesh()
    z = afile.load_function(mesh, 'z')
    adot = afile.load_function(mesh, 'adot')
    edge_field = afile.load_function(mesh, 'edge')

    adot.interpolate(10.*adot - 20.*edge_field)

    out = fd.File('mt_training/z.pvd')
    out.write(adot)

    """
    B = afile.load_function(mesh, 'B')
    adot = afile.load_function(mesh, 'adot')
    beta2 = afile.load_function(mesh, 'beta2')
    c = mesh.coordinates.vector().array()
    """