from dem_stitcher import stitch_dem
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
import math
import geopandas as gp
from dem_stitcher import stitch_dem
import rasterio
import glob
import gmsh
from scipy.special import expit
from scipy.ndimage import gaussian_filter 
from scipy.ndimage import distance_transform_edt
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib.colors as mcolors
import sys
import pandamesh as pm
from shapely import Polygon, to_geojson, from_geojson
from pathlib import Path
from shapely.geometry import MultiPolygon, Polygon
import firedrake as fd
from matplotlib import colors
import scipy


def get_mt_forests():

    forests_data = gp.read_file('summer/usa_forests/S_USA.AdministrativeForest.shp')
    
    states_data = gp.read_file('summer/states/cb_2018_us_state_500k.shp')
    state = states_data.loc[states_data['NAME'] == 'Montana']
    mt_forests = gp.overlay(forests_data, state, how='intersection')

    fig, ax = plt.subplots()
    mt_forests.plot(ax=ax, facecolor='none', edgecolor='red')
    state.plot(ax=ax, facecolor='none', edgecolor='blue')
    plt.show()

    forest_names = []
    geoms = []

    for idx, row in mt_forests.iterrows():
        k = 0 
        polys = row.geometry
        name = row['FORESTNAME'].split(' ')[0].split('-')[0]

        for p in  polys.geoms:
            if p.area > 0.1:
               geoms.append(p)
               forest_names.append(name)
               k += 1

    filtered = gp.GeoDataFrame(geometry=geoms, crs=mt_forests.crs)
    filtered['forest'] = forest_names
    filtered.to_file('summer/data/mt_forests.geojson')


def get_dems(buffer = 0.05):

    mt_forests = gp.read_file('summer/data/mt_forests.geojson')
    grouped = mt_forests.groupby('forest')

    for name, group in grouped:
        k = 0
        for idx, row in group.iterrows():
            name = row['forest']
            p = row.geometry
            p = p.buffer(buffer)
            bounds = list(p.bounds)
            bounds[0] -= buffer
            bounds[1] -= buffer
            bounds[2] += buffer
            bounds[3] += buffer

            X, g = stitch_dem(bounds,
                     dem_name='glo_90',  # Global Copernicus 30 meter resolution DEM
                     dst_ellipsoidal_height=False,
                     dst_area_or_point='Point')

            with rasterio.open(f'summer/data/{name}_{k}.tif', 'w', **g) as ds:
                ds.write(X, 1)
                ds.update_tags(AREA_OR_POINT='Point')

            k += 1


def reproject_dems(res = 100.):
    file_names = glob.glob('summer/data/*.tif')
    for file_name in file_names:
        X = rioxarray.open_rasterio(file_name)
        X = X.rio.reproject("EPSG:3857", resolution=res, resampling=Resampling.bilinear)
        X.rio.to_raster(file_name)


def write_meshes(res = 1.):
    pad = 100
    file_names = glob.glob('summer/data/*.tif')
    for file_name in file_names:
        name = file_name.split('/')[-1].split('.')[0]
        z = rioxarray.open_rasterio(file_name)[:,::2,::2]

        dy = z.x.max() - z.x.min()
        dx = z.y.max() - z.y.min()
        dx /= 1000.
        dy /= 1000.
        dx = dx.item()
        dy = dy.item()

        mask = xr.ones_like(z)
        mask[:,0:pad,:] = 0.
        mask[:,-pad:,:] = 0.
        mask[:,:,0:pad] = 0.
        mask[:,:,-pad:] = 0.
        mask = mask.data[0]
        #size = (1. - mask)*(2.5 - res) + res

        distance_transform = distance_transform_edt(mask == 0)
        signed_distance_field = np.where(mask != 1, -distance_transform, distance_transform)
        mask_field = 1.-2.0*expit(signed_distance_field / 40.)
        size = res + mask_field*(2.5 - res)

        x = np.linspace(0., dx, z.shape[1])
        y = np.linspace(0., dy, z.shape[2])
        xx, yy = np.meshgrid(y, x)
        size_interp = LinearNDInterpolator(list(zip(xx.flatten(), yy.flatten())), size[::-1,:].flatten())

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
        gmsh.write(f'summer/data/{name}_{int(res*1000)}.msh')

        # Launch the GUI to see the results:
        #if '-nopopup' not in sys.argv:
        #    gmsh.fltk.run()

        gmsh.finalize()


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

def write_data(name, res=500):

    mesh = fd.Mesh(f'summer/data/{name}_{res}.msh')
    V = fd.FunctionSpace(mesh, 'CG', 1)
    z_f = fd.Function(V, name='z')
    z_smooth_f = fd.Function(V, name='z_smooth')
    adot_f = fd.Function(V, name='adot')
    edge_f = fd.Function(V, name='edge')
    beta2_f = fd.Function(V, name='beta2')

    z = rioxarray.open_rasterio(f'summer/data/{name}.tif')
    dy = z.x.max() - z.x.min()
    dx = z.y.max() - z.y.min()
    dx /= 1000.
    dy /= 1000.
    dx = dx.item()
    dy = dy.item()
    z.data[0][np.isnan(z.data[0])] = 0.

    z_smooth = gaussian_filter(z.data[0], sigma = 75., mode='reflect')
    adot = -2.*(expit((z_smooth - z.data[0]) / 150.) - 0.5)

    pad = 225
    mask = xr.ones_like(z)
    mask[:,0:pad,:] = 0.
    mask[:,-pad:,:] = 0.
    mask[:,:,0:pad] = 0.
    mask[:,:,-pad:] = 0.
    mask = mask.data[0]


    distance_transform = distance_transform_edt(mask == 0)
    signed_distance_field = np.where(mask != 1, -distance_transform, distance_transform)
    edge_field = 1.-2.0*expit(signed_distance_field / 75.)
   
    beta2 = gen_field(mask.shape[0], mask.shape[1], 4000) #+ 0.25*(gen_field(mask.shape[0], mask.shape[1], 100) - 0.5)
    #beta2 = (adot_field/2. + 0.5)*beta2
    beta2 /= beta2.max()
    #beta2 = beta2**2

    norm = colors.TwoSlopeNorm(vmin=-1., vcenter=0., vmax=1.)
    plt.subplot(4,1,1)
    plt.imshow(z[0])
    plt.colorbar()

    plt.subplot(4,1,2)
    plt.imshow(adot, norm=norm, cmap='seismic')
    plt.colorbar()

    plt.subplot(4,1,3)
    plt.imshow(edge_field)
    plt.colorbar()

    plt.subplot(4,1,4)
    plt.imshow(beta2)
    plt.colorbar()
    plt.show()

    x = np.linspace(0., dx, z.shape[1])
    y = np.linspace(0., dy, z.shape[2])
    xx, yy = np.meshgrid(y, x)
    z_interp = LinearNDInterpolator(list(zip(xx.flatten(), yy.flatten())), z.data[0][::-1,:].flatten())
    z_smooth_interp = LinearNDInterpolator(list(zip(xx.flatten(), yy.flatten())), z_smooth[::-1,:].flatten())
    #adot_interp = LinearNDInterpolator(list(zip(xx.flatten(), yy.flatten())), adot[::-1,:].flatten())
    edge_interp = LinearNDInterpolator(list(zip(xx.flatten(), yy.flatten())), edge_field[::-1,:].flatten())
    beta2_interp = LinearNDInterpolator(list(zip(xx.flatten(), yy.flatten())), beta2[::-1,:].flatten())

    mesh_coords = mesh.coordinates.dat.data[:]
    z_mesh = z_interp(mesh_coords[:,0], mesh_coords[:,1])
    #adot_mesh = adot_interp((mesh_coords[:,0], mesh_coords[:,1]))
    edge_mesh = edge_interp(mesh_coords[:,0], mesh_coords[:,1])
    beta2_mesh = beta2_interp(mesh_coords[:,0], mesh_coords[:,1])
    z_smooth_mesh = z_smooth_interp(mesh_coords[:,0], mesh_coords[:,1])

    z_f.dat.data[:] = z_mesh
    #adot_f.dat.data[:] = adot_mesh
    edge_f.dat.data[:] = edge_mesh
    beta2_f.dat.data[:] = beta2_mesh
    z_smooth_f.dat.data[:] = z_smooth_mesh

    """
    plt.subplot(4,1,1)
    plt.scatter(mesh_coords[:,0], mesh_coords[:,1], c=z_mesh)
    plt.colorbar()

    plt.subplot(4,1,2)
    plt.scatter(mesh_coords[:,0], mesh_coords[:,1], c=adot_mesh)
    plt.colorbar()

    plt.subplot(4,1,3)
    plt.scatter(mesh_coords[:,0], mesh_coords[:,1], c=edge_mesh)
    plt.colorbar()

    plt.subplot(4,1,4)
    plt.scatter(mesh_coords[:,0], mesh_coords[:,1], c=beta2_mesh)
    plt.colorbar()

    plt.show()
    """

    with fd.CheckpointFile(f'summer/data/input_{name}_{res}.h5', 'w') as afile:
        afile.save_mesh(mesh)
        afile.save_function(z_f)
        #afile.save_function(adot_f)
        afile.save_function(edge_f)
        afile.save_function(beta2_f)
        afile.save_function(z_smooth_f)


   


#write_meshes(1.)

write_data('Beaverhead_0', 500) 
write_data('Beaverhead_0', 1000) 
write_data('Beaverhead_1', 500) 
write_data('Beaverhead_1', 1000) 
write_data('Beaverhead_2', 500) 
write_data('Beaverhead_2', 1000) 

