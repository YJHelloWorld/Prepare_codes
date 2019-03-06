import numpy as np
import healpy as hp
import matplotlib as mpl
mpl.use('PDF')
import matplotlib.pyplot as plt
import pysm
from pysm.nominal import models
from pysm.common import convert_units
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

nu = np.array([95.]); Nf = len(nu)
coefficients = convert_units("uK_RJ", "uK_RJ", nu)
def convert_unit(map):
    for i in range(0,Nf):
        map[i] = map[i]*coefficients[i]
    return map
nside = 128
f1_config = models("f1", nside)	
Cl = []
fig, ax = plt.subplots();k = 0
for i in range(1):		
    for s in range(1,4): #3
        s1_config = models("s%s"%s, nside)
		
	for d in range(1,8): #8
	    d1_config = models("d%s"%d, nside)
		    
	    for a in range(2,3): #2
		a1_config = models("a%s"%a, nside)
#		c1_config = models("c1", 128)
	
		sky_config = {
		                'synchrotron' : s1_config,
		                'dust' : d1_config,
		                'freefree' : f1_config,
		                'ame' : a1_config,
		            }
		sky = pysm.Sky(sky_config);
		total = sky.signal()(nu);
		total = convert_unit(total)  
		cls = hp.anafast(total); Cl.append(cls[2])
		ell = np.arange(len(cls[2][2:]))
		plt.loglog(ell+2, (ell+2)*(ell+3)/2/np.pi*cls[2][2:], label='%s'%k)
	        k += 1
plt.xlabel(r'$\ell$'); plt.ylabel(r'$\ell(\ell+1)C_\ell/2\pi(\mu K^2_{RJ})$')
#plt.legend();
plt.show()
plt.savefig('foregrounds_tensor.pdf', format = 'pdf')	
np.savetxt('Cl_of_foreground.txt', Cl)
