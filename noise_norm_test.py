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
#os.system('rm -rf /smc/jianyao/datasets/image_matrix/2_channels/train; mkdir /smc/jianyao/datasets/image_matrix/2_channels/train')
import itchat
import threading

nu = np.array([95., 150., 353.]) 
#nu = np.array([70., 95., 150.])
#nu = np.array([95., 150.,217., 353.])
#nu = np.array([95, 150])
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

#Nl = (0.066**2,0.065**2,0.063**2,0.028**2,0.015**2,0.023**2,0.068**2)
#Nl = [0.028**2,0.015**2,0.068**2]
Nl = [1e-5, 1e-5, 1e-5]
global k
Nf = len(nu) ; k=0; pp = 2    #polarization parameter 
names = ['intensity', 'Q_map', 'U_map']
cl_real = hp.anafast(cmb)  #TT, EE, BB, TE, EB, TB

@itchat.msg_register(itchat.content.TEXT)
def text_reply(msg):
    if msg['Text'] == 'epoch':
	itchat.send('k =%s '%k)
#itchat.auto_login(enableCmdQR=2,hotReload=True)
#threading.Thread(target = itchat.run).start()
Fg = [] # collection for fg...
f1_config = models("f1", nside)	
for i in range(1):		
    for s in range(1,3): #3
        s1_config = models("s%s"%s, nside)
		
	for d in range(1,8): #7
	    d1_config = models("d%s"%d, nside)
		    
	    for a in range(1,2): #2
		a1_config = models("a%s"%a, nside)
		c1_config = models("c1", 128)
	
		sky_config = {
		                'synchrotron' : s1_config,
		                'dust' : d1_config,
		                'freefree' : f1_config,
		                'ame' : a1_config,
		            }
		sky = pysm.Sky(sky_config);
		total = sky.signal()(nu) 
###
#		sync = sky.synchrotron(nu)
#		total = total + sync*1e3
### to let the sync with dust
		total = convert_unit(total)
		cmb_all = []; total_all = []
		# multiply every cl with a factor randomly generated between [0.5,2], 07_02 21:54
#		for l in range(3*nside):
#		    cl[l] = cl_real[l]*factor[l]
##		coeff_cl = np.random.uniform(100,5000,3*nside)
##                ell = np.arange(384); ell[0] = 1
##               cl = coeff_cl*2*np.pi/ell/(ell+1)
#               cmb_map = hp.synfast(cl,nside)
#		factor = np.random.uniform(0.5,10,3*nside)
#		cl = cl_real*factor
		cmb_map = hp.synfast(cl_real, nside, new = True)
#		cmb_map = np.random.uniform(-600,700,12*nside**2)
	 	k += 1
		print k
		white_noise = np.ones((Nf, 12*nside**2))
		for i in range(Nf):
	            nl = np.ones(len(cl_real[0])); nl[:] = Nl[i]
		    white_noise[i][:]= hp.synfast(nl,nside)[:]
###
		NTF = Nf -1    ###   normalize to which frequency
###		mean_Q = np.mean(total[NTF][1]+cmb_map[1]+white_noise[NTF]); mean_U = np.mean(total[NTF][2]+cmb_map[2]+ white_noise[NTF])
		std_Q = np.std(total[NTF][1]+cmb_map[1]+white_noise[NTF]); std_U = np.std(total[NTF][2]+cmb_map[2]+white_noise[NTF])
		fnf = np.sqrt(std_Q**2 + std_U**2)  #fnf: fiducial norm factor
		fg = []; fg.append(fnf)
#		Q_U = [0, std_Q, std_U]; Q_U_mean = [0, mean_Q, mean_U]
####
		with file('/smc/jianyao/datasets/image_matrix/noise_ali/test/QU_7_fre_%s.txt'%(k), 'w') as outfile:
			
		    for fre in range(Nf):
			total_with_cmb_Q = cmb_map[1] + total[fre][1] + white_noise[fre]
                        total_with_cmb_U = cmb_map[2] + total[fre][2] + white_noise[fre]
                        nf = np.sqrt((np.std(total_with_cmb_Q)**2 + np.std(total_with_cmb_U)**2)/2)   #nf: norm factor
			fg.append(nf)
		        for j in range(1,3):
			    
	#              
			    cmb_norm = cmb_map[j]/nf*fnf ##-np.mean(total_with_cmb))/np.std(total_with_cmb)*Q_U[j]
                            cmb_matrix = cmb_norm[pixels]
                            total_norm = (cmb_map[j]+ total[fre][j] + white_noise[fre] )/nf*fnf ##- np.mean(total_with_cmb))/np.std(total_with_cmb)*Q_U[j]
                            total_matrix = total_norm[pixels]

####
	#		    cmb_matrix = cmb_map[j][pixels]
	#	            total_matrix = total[fre][j][pixels] + cmb_matrix
####
		            combine = np.column_stack((cmb_matrix, total_matrix))
		            outfile.write('#Array shape:{0}\n'.format(combine.shape))
	#                np.savetxt('/home/jianyao/pytorch-CycleGAN-and-pix2pix/datasets/image_matrix/train/totals%s.txt'%(i+687),combine, fmt = '%.2f')
			    np.savetxt(outfile, combine, fmt = '%.4f')
			    outfile.write('#New dimension\n')
			if fre == (NTF-1):
			    break
		    for j in range(1,3):
			cmb_norm = (cmb_map[j])#-Q_U_mean[j])/Q_U[j]
                        cmb_matrix = cmb_norm[pixels]
                        total_norm = (total[NTF][j] +cmb[j] + white_noise[NTF])# - Q_U_mean[j])/Q_U[j]
                        total_matrix = total_norm[pixels]

			combine = np.column_stack((cmb_matrix, total_matrix))
                        outfile.write('#Array shape:{0}\n'.format(combine.shape))
        #                np.savetxt('/home/jianyao/pytorch-CycleGAN-and-pix2pix/datasets/image_matrix/train/totals%s.txt'%(i+687),combine, fmt = '%.2f')
                        np.savetxt(outfile, combine, fmt = '%.4f')
                        outfile.write('#New dimension\n')
		Fg.append(fg)
np.savetxt('/smc/jianyao/checkpoints/noise_3_fres_norm_new/norm_factors.txt', Fg)
'''				
		with file('/smc/jianyao/datasets/image_matrix/train/%s_%d.txt'%(names[2], k), 'w') as outfile:

                    for j in range(7):
        #                 nl = np.ones_like(cl); nl[:] = Nl[j]
        #                 noise = hp.synfast(nl,64)
        #                 
			cmb_matrix = cmb_map[2][pixels]
                        total_matrix = total[j][2][pixels]
                        combine = np.column_stack((cmb_matrix, total_matrix))
                        outfile.write('#Array shape:{0}\n'.format(combine.shape))
        #                np.savetxt('/home/jianyao/pytorch-CycleGAN-and-pix2pix/datasets/image_matrix/train/totals%s.txt'%(i+687),combine, fmt = '%.2f')
                        np.savetxt(outfile, combine, fmt = '%.4f')
                        outfile.write('#New dimension\n')

'''

#os.system('python ../test.py --dataroot /smc/jianyao/datasets/image_matrix/noise_ali --name /smc/jianyao/checkpoints/noise_ali --model pix2pix --which_model_netG unet_256 --which_direction BtoA --dataset_mode aligned --no_lsgan --norm batch --pool_size 0 --loadSize 256 --fineSize 256 --gpu_ids 0,1 --input_nc 6 --output_nc 2 --display_freq 1 --batchSize 32 --save_epoch_freq 100 --display_id 0 --niter 300 --niter_decay 300')
