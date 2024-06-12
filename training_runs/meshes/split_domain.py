import numpy as np
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt

b = rioxarray.open_rasterio('data/B.tif')
_, ny, nx = b.shape
size = 2000
nx = int(np.floor(nx / size))
ny = int(np.floor(ny / size))
print(nx, ny)

k = 0
l = 0
ks = set([1,3,6,7,9,10,12,14])
for i in range(nx):
    for j in range(ny):
        print(k)
        if k in ks:
            b_ij = b[:, j*size:(j+1)*size, i*size:(i+1)*size]
            b_ij.rio.to_raster(f'data/z_{l}.tif')
            plt.imshow(b_ij[0])
            plt.colorbar()
            plt.show()
            l += 1


        k += 1

 

