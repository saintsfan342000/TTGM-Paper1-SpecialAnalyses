import numpy as n
L = n.genfromtxt
abs=n.abs
import matplotlib.pyplot as p
import os
from sys import argv
from scipy.signal import detrend
import figfun as ff
from pandas import read_excel

worthless, expt, FS, SS= argv
expt = int(expt)
FS = int(FS)
SS = int(SS)

#########
#Max.dat
#   '[0]Stage [1]Time [2]AxSts [3]ShSts [4]NEx [5]NEy [6]Gamma [7]F11-1 [8]F22-1 [9]atan(F12/F22) [10]epeq [11]AramX [12]AramY'
#mean.dat
#   [0]Stage [1]Time [2]NumPtsPassed [3]AxSts [4]ShSts [5]NEx [6]NEy [7]Gamma [8]F11-1 [9]F22-1 [10]atan(F12/F22) [11]epeq'
#like_Scott
#   [0]Stage [1]Time [2]SizeAveragingZone(in) [3]AxSts [4]ShSts [5]NEx [6]NEy [7]Gamma [8]F11-1 [9]F22-1 [10]atan(F12/F22) [11]epeq'
# profStgs
#   'Stages at which profiles were generated'
# profUr
#   'First Row Stage number.  Second row begin data.\n[0]Ycoord [1:] Stage Ur/Ro'
#MaxPt.dat
#   [0]Stage [1]Time [2]AxSts [3]ShSts [4]NEx [5]NEy [6]Gamma [7]F11-1 [8]F22-1 [9]atan(F12/F22) [10]epeq'

pwd = os.getcwd()
os.chdir('../../TTGM-{}_FS{}SS{}'.format(expt, FS, SS))

STF = L('STF.dat',delimiter=',')
dmax = L('max.dat',delimiter=',')
dmaxPt = L('MaxPt.dat',delimiter=',')
dmean = L('mean.dat',delimiter=',')
dscot = L('like_Scott.dat',delimiter=',')
DR = L('disp-rot.dat',delimiter=',')
#'[0]Stage [1]Time [2]AxSts [3]ShSts [4]Delta/L [5]Phi. Lg = {:.6f} inch'
profStg = L('prof_stages.dat',delimiter=',').astype(int)
profLEp = L('StrainProfiles.dat',delimiter=',')[1:]
profUr = L('RadialContraction.dat',delimiter=',')[1:]

##################################################
# Figure 4 - Radial contraction profile thru x = 0
##################################################

os.chdir(pwd)

profUr = profUr[~n.any(n.isnan(profUr),axis=1), :]  # Detrend the data!
profUr[:,1:] = detrend(profUr[:,1:],axis=0)
profUr[:,1:] -= n.nanmax( profUr[:,1:], axis=0 )
D = n.c_[2*profUr[:,0]/0.62,profUr[:,1:]]
D = D[ (D[:,0]>=-1) & (D[:,0]<=1) ]
n.savetxt('TTGM-{}_RadContraction.dat'.format(expt), fmt='%.6f', delimiter=',',
          X = D,
          header='# [0]2*Y/0.62, [1:] Stage Ur/Ro')

p.style.use('mysty')
fig4 = p.figure(4,facecolor='w',figsize=(12,6) )
p.gcf().add_axes([.12,.12,.8,.78])    
p.plot(D[:,0], D[:,1:],lw=1.5)
#p.gcf().gca().set_ylim(bottom=0)
p.xlabel('2y$_{\\mathsf{o}}$/L$_{\\mathsf{g}}$')
p.ylabel('$\\frac{\\mathsf{u}_{\\mathsf{r}}}{\\mathsf{R}_{\\mathsf{o}}}$')
p.gca().set_xlim([-1,1])
p.grid(True,which='both',axis='y',linestyle='--',alpha=0.5)          
p.show()