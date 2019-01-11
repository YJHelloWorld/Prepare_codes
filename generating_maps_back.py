
import pysm
from pysm.nominal import models
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from pysm.common import convert_units
import matplotlib as mpl
import numpy.ma as ma

from NPTFit import create_mask as cm

nu = np.array([30., 44., 70., 95., 150., 217., 353.]) 
coefficients = convert_units("uK_RJ", "uK_CMB", nu)
def convert_unit(map):
    for i in range(0,Nf):
        map[i] = map[i]*coefficients[i]
    return map

c1_config = models("c1", 128)
sky_config = {'cmb' : c1_config}
sky = pysm.Sky(sky_config)
cmb = sky.signal()(30)
cmb = cmb*convert_units("uK_RJ", "uK_CMB", 30)

xsize = 256; ysize = 256
nside = 128
phi = np.radians(np.linspace(5,85, xsize))   #0-180 start from the North Pole ,latitude
theta = np.radians(np.linspace(-60, 60, ysize)) # -180-180 start from the right , longtitude
PHI, THETA = np.meshgrid(phi, theta)
pixels = hp.ang2pix(nside,PHI, THETA)

cl = hp.anafast(cmb[0])
nu = np.array([30., 44., 70., 95., 150., 217., 353.]) 
Nl = (0.066**2,0.065**2,0.063**2,0.028**2,0.015**2,0.023**2,0.068**2)
Nf = 7 ; i=0
f1_config = models("f1", nside)			
for s in range(1,2): #3
    s1_config = models("s%s"%s, nside)
    
    for d in range(1,2): #7
        d1_config = models("d%s"%d, nside)
        
        for a in range(1,2): #2
            a1_config = models("a%s"%a, nside)
    
            sky_config = {
                    'synchrotron' : s1_config,
                    'dust' : d1_config,
                    'freefree' : f1_config,
                    'ame' : a1_config
                }
            sky = pysm.Sky(sky_config)
            total = sky.signal()(nu) 
            total = convert_unit(total)
            cmb_all = []; total_all = []
            cmb_map = hp.synfast(2*cl, nside)
            cmb_matrix = cmb_map[pixels]
	    i =+ 1 
	    with file('/smc/jianyao/datasets/image_matrix/train/totals%s.txt'%(i), 'w') as outfile:
			
            	for j in range(7):
#                 nl = np.ones_like(cl); nl[:] = Nl[j]
#                 noise = hp.synfast(nl,64)
#                 
                    total_matrix = total[j][0][pixels] + cmb_matrix
                    combine = np.column_stack((cmb_matrix, total_matrix))
		    outfile.write('#Array shape:{0}\n'.format(combine.shape))
#                np.savetxt('/home/jianyao/pytorch-CycleGAN-and-pix2pix/datasets/image_matrix/train/totals%s.txt'%(i+687),combine, fmt = '%.2f')
		    np.savetxt(outfile, combine, fmt = '%.4f')
    		    outfile.write('#New dimension\n')
