import numpy as n
L = n.genfromtxt
abs=n.abs
import matplotlib.pyplot as p
import os
from sys import argv
from scipy.signal import detrend
from scipy.interpolate import interp1d
import figfun as ff
from tqdm import trange
#from pandas import read_excel

# Increm strains
def increm_strains(A,B):
    '''
    Requires A and B be (nx4), with cols corresponding to F00, F01, F10, F11
    '''
    de00 = (2*((A[:,0] - B[:,0])*(A[:,3] + B[:,3]) - (A[:,1] - B[:,1])*(A[:,2] + B[:,2]))/
            ((A[:,0] + B[:,0])*(A[:,3] + B[:,3]) - (A[:,1] + B[:,1])*(A[:,2] + B[:,2]))
           )
    de01 = ((-(A[:,0] - B[:,0])*(A[:,1] + B[:,1]) + (A[:,0] + B[:,0])*(A[:,1] - B[:,1]) + 
            (A[:,2] - B[:,2])*(A[:,3] + B[:,3]) - (A[:,2] + B[:,2])*(A[:,3] - B[:,3]))/
            ((A[:,0] + B[:,0])*(A[:,3] + B[:,3]) - (A[:,1] + B[:,1])*(A[:,2] + B[:,2]))
           )
    de11 = (2*((A[:,0] + B[:,0])*(A[:,3] - B[:,3]) - (A[:,1] + B[:,1])*(A[:,2] - B[:,2]))/
            ((A[:,0] + B[:,0])*(A[:,3] + B[:,3]) - (A[:,1] + B[:,1])*(A[:,2] + B[:,2]))
           )
    return de00, de01, de11

def increm_rot(A,B):
    '''
    Requires A and B be (nx4), with cols corresponding to F00, F01, F10, F11
    This is the 01 component of the incremental spin tensor
    '''
    dr = ((-(A[:,0] - B[:,0])*(A[:,1] + B[:,1]) + (A[:,0] + B[:,0])*(A[:,1] - B[:,1]) -
           (A[:,2] - B[:,2])*(A[:,3] + B[:,3]) + (A[:,2] + B[:,2])*(A[:,3] - B[:,3]))/
           ((A[:,0] + B[:,0])*(A[:,3] + B[:,3]) - (A[:,1] + B[:,1])*(A[:,2] + B[:,2]))
         )
    return -dr  

def deeq(E,sig):
    '''
    E must be  nx3, columns corresponding to de00, de01, de11
    sig must be a list or tuple of (sig00, sig01, sig11, sigeq)
    '''
    return (E[:,0]*sig[0]+E[:,2]*sig[2]-2*E[:,1]*sig[1])/sig[3]

worthless, expt, FS, SS= argv
#expt, FS, SS = 7, 19, 6
expt = int(expt)
FS = int(FS)
SS = int(SS)

path = '../../TTGM-{}_FS{}SS{}'.format(expt, FS, SS)

STF = L('{}/STF.dat'.format(path),delimiter=',')
last = int(STF[-1,0])
profStg = L('{}/prof_stages.dat'.format(path),delimiter=',').astype(int)
key = n.genfromtxt('../../ExptSummary.dat', delimiter=',')
thick = key[ key[:,0] == int(expt), 6 ].ravel()

sig00 = STF[:,2]/2  # Hoop sts (assumed 1/2 axial)
sig11 = STF[:,2]   # Ax sts
sig01 = STF[:,3]   # Sh sts
sigvm = n.sqrt(sig11**2 + sig00**2 - sig00*sig11 + 3*sig01**2)


# [0]Index_x [1]Index_y [2,3,4]Undef_X,Y,Z inches 
# [5,6,7]Def_X,Y,Z inches [8,9,10,11]DefGrad (11 12 21 22) *)
# [12,13,14] e00, e01, e11
# [15,16,17] eeqVM, eeqH8, eeqAnis
D = n.load('{}/IncrementalAnalysis/NewFilterPassingPoints_3med.npy'.format(path))
D = D.take(profStg, axis=0)
locmax = D[-1, :, 15].argmax()
maxi = D[-1, locmax, 0]
maxycoord = D[-1, locmax, 3]

for i in trange(1, last+1):
    D = n.load('{}/AramisBinary/TTGM-{}_FS{}SS{}_{}.npy'.format(path,expt,FS,SS,i))
    rng = (D[:,0] == maxi) & (D[:,3]>=maxycoord-6*thick) & (D[:,3]<=maxycoord+6*thick)
    D = D[rng]
    D[:,3] -= maxycoord
    if i == 1:
        rng = (D[:,3]>=maxycoord-5*thick) & (D[:,3]<=maxycoord+5*thick)
        Dtemp = D[rng]
        yspace = n.linspace(Dtemp[:,3].min(), Dtemp[:,3].max(), Dtemp.shape[0])
        de = n.zeros((len(yspace), last+1))
    AFint = interp1d(D[:,3], D[:,8:], axis=0).__call__(yspace)
    if i == 1:
        BFint = n.ones_like(AFint) * [1,0,0,1]
    de00, de01, de11 = increm_strains(AFint, BFint)
    deq = deeq(n.c_[de00, de01, de11], (sig00[i], sig01[i], sig11[i], sigvm[i]))
    de[:,i] = de[:,i-1]+deq
    BFint = AFint.copy()

for i in profStg:
    p.plot(yspace/thick, de[:,i])
p.show()


n.savetxt('TTGM-{}_IncrementalProfile.dat'.format(expt), fmt='%.6f', delimiter=',',
          X = n.c_[yspace/thick, de.take(profStg, axis=1)],
              header='First Col:  yo/to.  Next 10 cols are the 10 profStgs'
          )

n.save('TTGM-{}_IncrementalProfile_AllStage.npy'.format(expt), de)

