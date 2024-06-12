import numpy as np
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.special import expit
from scipy.ndimage import gaussian_filter 
import matplotlib.colors as mcolors
import gmsh
import sys
from scipy.interpolate import LinearNDInterpolator

def create_mesh(k, name):
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
    z_smooth = gaussian_filter(z, 100.)
    adot_field = 2.*(expit((z - z_smooth) / 500.) - 0.5)

    distance_transform = distance_transform_edt(mask == 0)
    signed_distance_field = np.where(mask != 1, -distance_transform, distance_transform)
    mask_field = 1.-2.0*expit(signed_distance_field / 30.)
    print(mask_field.min(), mask_field.max())

    #plt.imshow(mask_field)
    #plt.colorbar()
    #plt.show()


    adot = 20.*adot_field #- 40.*mask_field    
    norm = mcolors.TwoSlopeNorm(vmin=adot.min(), vcenter=0., vmax=adot.max())
    adot = adot*(1-mask_field) + mask_field*(-20)
   
    #plt.imshow(adot, norm=norm, cmap='seismic_r')
    #plt.colorbar()
    #plt.show()
    

    ### Mesh
    x = np.linspace(0., dx, z.shape[0])
    y = np.linspace(0., dy, z.shape[1])
    xx, yy = np.meshgrid(y, x)

    mask_field = 0.5 + 2.5*mask_field
    size_interp = LinearNDInterpolator(list(zip(xx.flatten(), yy.flatten())), mask_field[::-1,:].flatten())

    gmsh.initialize(sys.argv)
    gmsh.model.add("t10")

    # Let's create a simple rectangular geometry:
    lc = .15
    eps = 0.1
    gmsh.model.geo.addPoint(eps, eps, 0, lc, 1)
    gmsh.model.geo.addPoint(dy-eps, eps, 0, lc, 2)
    gmsh.model.geo.addPoint(dy-eps, dx-eps, 0, lc, 3)
    gmsh.model.geo.addPoint(eps, dx-eps, 0, lc, 4)

    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)

    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 5)
    gmsh.model.geo.addPlaneSurface([5], 6)

    gmsh.model.geo.synchronize()


    # The API also allows to set a global mesh size callback, which is called each
    # time the mesh size is queried
    def meshSizeCallback(dim, tag, x, y, z, lc):
        return min(lc, size_interp(x,y))


    gmsh.model.mesh.setSizeCallback(meshSizeCallback)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.option.setNumber("Mesh.Algorithm", 5)

    gmsh.model.mesh.generate(2)
    gmsh.write(name)

    # Launch the GUI to see the results:
    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()

    gmsh.finalize()

#for i in range(7):
i = 0
create_mesh(i, f'data/mesh_500.msh')