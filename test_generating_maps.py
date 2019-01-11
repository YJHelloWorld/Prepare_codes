import os
import pysm
from pysm.nominal import models
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from pysm.common import convert_units
import matplotlib as mpl
import numpy.ma as ma

from NPTFit import create_mask as cm

os.system('rm -rf /smc/jianyao/datasets/image_matrix/test; mkdir /smc/jianyao/datasets/image_matrix/test')

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

cl_real = hp.anafast(cmb[0])
cl_camb = np.loadtxt('cl_camb.txt')[0:3*nside]
nu = np.array([30., 44., 70., 95., 150., 217., 353.]) 
Nl = (0.066**2,0.065**2,0.063**2,0.028**2,0.015**2,0.023**2,0.068**2)
Nf = 7 ; k=0; coeff = 1
f1_config = models("f1", nside)	
for i in range(1):		
    for s in range(1,4): #3
        s1_config = models("s%s"%s, nside)
		
	for d in range(1,8): #7
	    d1_config = models("d%s"%d, nside)
		    
	    for a in range(1,3): #2
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
#		factor = np.random.uniform(19.0,21.0,3*nside)
#		factor = np.ones(3*nside)*5.25
#               cl = cl_real*factor
#		coeff_cl = np.random.uniform(100,5000,3*nside)
                ell = np.arange(384); ell[0] = 1
                cl = cl_camb*2*np.pi/ell/(ell+1)
		
#		cl =cl_real*4
                cmb_map = hp.synfast(cl,nside)

#		coeff_cl = np.ones(384)*500 #+ np.random.random(384)
#		ell = np.arange(384); ell[0] = 1
#		cl = coeff_cl*2*np.pi/ell/(ell+1)
#		cmb_map = hp.synfast(cl,nside)
#		cmb_map = np.random.uniform(-600,700,12*nside**2)
		cmb_matrix = cmb_map[pixels]
	 	k += 1
		with file('/smc/jianyao/datasets/image_matrix/test/camb_cl_%s.txt'%(k), 'w') as outfile:
			
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
os.system(' python ../test.py --dataroot /smc/jianyao/datasets/image_matrix --name /smc/jianyao/checkpoints/random_and_normal_cl_0713 --model pix2pix --which_model_netG unet_256 --which_direction BtoA --dataset_mode aligned --norm batch --input_nc 7 --output_nc 1')
