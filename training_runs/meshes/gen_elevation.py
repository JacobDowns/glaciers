import numpy as np
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.special import expit
from scipy.ndimage import gaussian_filter 
import matplotlib.colors as mcolors

for k in range(1,9):
    z = rioxarray.open_rasterio(f'data/z_{k}.tif')  

    mask = xr.ones_like(z)
    pad = 200
    mask[:,0:pad,:] = 0.
    mask[:,-pad:,:] = 0.
    mask[:,:,0:pad] = 0.
    mask[:,:,-pad:] = 0.
    mask = mask.data[0]
    
    z = z.data[0]
    z_smooth = gaussian_filter(z, 100.)
    adot_field = 2.*(expit((z - z_smooth) / 500.) - 0.5)

  
    #norm = mcolors.TwoSlopeNorm(vmin=adot.min(), vcenter=0., vmax=adot.max())

    distance_transform = distance_transform_edt(mask == 0)
    signed_distance_field = np.where(mask != 1, -distance_transform, distance_transform)
    mask_field = 1.-2.0*expit(signed_distance_field / 100.)

    adot = 20.*adot_field #- 40.*mask_field    
    norm = mcolors.TwoSlopeNorm(vmin=adot.min(), vcenter=0., vmax=adot.max())
    adot -= 30*mask_field 

    plt.imshow(adot, norm=norm, cmap='seismic_r')
    plt.colorbar()
    plt.show()
    quit()
    adot = z - z.mean()
    znew = z.data[0]*mask_field + z.data[0].min()*(1-mask_field)

    plt.imshow(znew)
    plt.colorbar()
    plt.show()

    quit()