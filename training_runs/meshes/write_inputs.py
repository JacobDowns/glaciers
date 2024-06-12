import numpy as np
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.special import expit
from scipy.ndimage import gaussian_filter 
import matplotlib.colors as mcolors
import sys
from scipy.interpolate import LinearNDInterpolator
import firedrake as fd
import scipy


def gen_field(nx, ny, correlation_scale):

    # Create the smoothing kernel
    x = np.arange(-correlation_scale, correlation_scale)
    y = np.arange(-correlation_scale, correlation_scale)
    X, Y = np.meshgrid(x, y)
    dist = np.sqrt(X*X + Y*Y)
    filter_kernel = np.exp(-dist**2/(2*correlation_scale))

    # Generate random noise and smooth it
    noise = np.random.randn(nx, ny) 
    z = scipy.signal.fftconvolve(noise, filter_kernel, mode='same')
        
    # Normalize so its in 0-1 range
    z -= z.min()
    z /= z.max()

    return z


def write_inputs(k, res=500):

    print(k)
    z = rioxarray.open_rasterio(f'data/z_{k}.tif')[:, ::2, ::2]  

    dx = z.x.max() - z.x.min()
    dy = z.y.max() - z.y.min()
    dx /= 1000.
    dy /= 1000.

    mask = xr.ones_like(z)
    pad = 125
    mask[:,0:pad,:] = 0.
    mask[:,-pad:,:] = 0.
    mask[:,:,0:pad] = 0.
    mask[:,:,-pad:] = 0.
    mask = mask.data[0]
    
    z = z.data[0]
    z_smooth = gaussian_filter(z, 250.)
    adot_field = 2.*(expit((z - z_smooth) / 600.) - 0.5)

    distance_transform = distance_transform_edt(mask == 0)
    signed_distance_field = np.where(mask != 1, -distance_transform, distance_transform)
    mask_field = 1.-2.0*expit(signed_distance_field / 30.)


    adot = 20.*adot_field #- 40.*mask_field    
    norm = mcolors.TwoSlopeNorm(vmin=adot.min(), vcenter=0., vmax=adot.max())
    adot = adot*(1-mask_field) + mask_field*(-20)
   
    plt.imshow(adot, norm=norm, cmap='seismic_r')
    plt.colorbar()
    plt.show()
    

    beta2 = gen_field(z.shape[0], z.shape[1], 4000) + 0.25*(gen_field(z.shape[0], z.shape[1], 100) - 0.5)
    #beta2 = (adot_field/2. + 0.5)*beta2
    beta2 /= beta2.max()
    beta2 = beta2**2
    plt.imshow(beta2**2)
    plt.colorbar()
    plt.show()

    ### Mesh
    x = np.linspace(0., dx, z.shape[0])
    y = np.linspace(0., dy, z.shape[1])
    xx, yy = np.meshgrid(y, x)

    B_interp = LinearNDInterpolator(list(zip(xx.flatten(), yy.flatten())), z[::-1,:].flatten())
    adot_interp = LinearNDInterpolator(list(zip(xx.flatten(), yy.flatten())), adot[::-1,:].flatten())
    beta2_interp = LinearNDInterpolator(list(zip(xx.flatten(), yy.flatten())), beta2[::-1,:].flatten())

    mesh = fd.Mesh(f'data/mesh_500.msh')

    V_cg = fd.FunctionSpace(mesh, 'CG', 1)
    vx, vy = fd.SpatialCoordinate(mesh)
    vx = fd.project(vx, V_cg).vector().array()
    vy = fd.project(vy, V_cg).vector().array()
    coords = np.c_[vx, vy]

    B_f = fd.Function(V_cg, name = 'B')
    adot_f = fd.Function(V_cg, name='adot')
    beta2_f = fd.Function(V_cg, name='beta2')
    H_f = fd.Function(V_cg, name='H')

    #B_f.dat.data[:] = B_interp(coords[:,0], coords[:,1])
    #adot_f.dat.data[:] = adot_interp(coords[:,0], coords[:,1])
    #beta2_f.dat.data[:] = beta2_interp(coords[:,0], coords[:,1])


    B_file = fd.File(f'data/B.pvd')
    B_file.write(B_f, idx=0)

    print(B_f.dat.data)

    adot_file = fd.File(f'data/adot.pvd')
    adot_file.write(adot_f, idx=0)

    beta2_file = fd.File(f'data/beta2.pvd')
    beta2_file.write(beta2_f, idx=0)

    quit()
    with fd.CheckpointFile(f'data/input_{k}_{res}.h5', 'w') as afile:
        afile.save_mesh(mesh)
        afile.save_function(B_f)
        afile.save_function(beta2_f)
        afile.save_function(adot_f)
        afile.save_function(H_f)

   
i = 0
write_inputs(i)