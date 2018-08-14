import numpy as n
from numpy import array, nanmean, nanstd, sqrt
from numpy import hstack, vstack, dstack
n.set_printoptions(linewidth=300,formatter={'float_kind':lambda x:'{:g}'.format(x)})
import matplotlib.pyplot as p
import figfun as f
import os
from tqdm import tqdm, trange
from sys import argv

'''
Just calculates avg. stn in a Site B zone
by averaging F for all points in that zone
'''

def increm_strains(A00,A01,A10,A11,B00,B01,B10,B11):
    de00 = (2*((A00 - B00)*(A11 + B11) - (A01 - B01)*(A10 + B10))/
            ((A00 + B00)*(A11 + B11) - (A01 + B01)*(A10 + B10))
           )
    de01 = ((-(A00 - B00)*(A01 + B01) + (A00 + B00)*(A01 - B01) + 
            (A10 - B10)*(A11 + B11) - (A10 + B10)*(A11 - B11))/
            ((A00 + B00)*(A11 + B11) - (A01 + B01)*(A10 + B10))
           )
    de11 = (2*((A00 + B00)*(A11 - B11) - (A01 + B01)*(A10 - B10))/
            ((A00 + B00)*(A11 + B11) - (A01 + B01)*(A10 + B10))
           )
    if type(de00) is n.ndarray:
        return de00.mean(), de01.mean(), de11.mean()
    else:
        return de00, de01, de11

def deeq(de00, de01, de11, sig00, sig01, sig11, sigeq):
    return (sig00*de00 + sig11*de11 - 2*sig01*de01)/sigeq

from numpy import sqrt, log

def Eig2x2(a,b,c,d):
    '''
    Mx must be [[a,b],[c,d]]
    '''
    return (a + d - sqrt(a**2 + 4*b*c - 2*a*d + d**2))/2, (a + d + sqrt(a**2 + 4*b*c - 2*a*d + d**2))/2

def LEp(F00, F01, F10, F11, rtnU=False):
    U00 = (sqrt(F00**2 + F01**2 + F10**2 + F11**2 - sqrt(((F01 + F10)**2 + (F00 - F11)**2)*
           ((F01 - F10)**2 + (F00 + F11)**2)))*(-F00**2 + F01**2 - F10**2 + F11**2 + 
        sqrt(((F01 + F10)**2 + (F00 - F11)**2)*((F01 - F10)**2 + (F00 + F11)**2))) + 
      (F00**2 - F01**2 + F10**2 - F11**2 + sqrt(((F01 + F10)**2 + (F00 - F11)**2)*
          ((F01 - F10)**2 + (F00 + F11)**2)))*sqrt(F00**2 + F01**2 + F10**2 + F11**2 + 
         sqrt(((F01 + F10)**2 + (F00 - F11)**2)*((F01 - F10)**2 + (F00 + F11)**2))))/(
     (2*sqrt(2)*sqrt(-4*(F01*F10 - F00*F11)**2 + (F00**2 + F01**2 + F10**2 + F11**2)**2)))


    U01 = (sqrt(2)*(F00*F01 + F10*F11))/(
     (sqrt(F00**2 + F01**2 + F10**2 + F11**2 - sqrt(((F01 + F10)**2 + (F00 - F11)**2)*
          ((F01 - F10)**2 + (F00 + F11)**2))) + 
      sqrt(F00**2 + F01**2 + F10**2 + F11**2 + sqrt(((F01 + F10)**2 + (F00 - F11)**2)*
          ((F01 - F10)**2 + (F00 + F11)**2)))))


    U11 = (sqrt(F00**2 + F01**2 + F10**2 + F11**2 - sqrt(((F01 + F10)**2 + (F00 - F11)**2)*
           ((F01 - F10)**2 + (F00 + F11)**2)))*(F00**2 - F01**2 + F10**2 - F11**2 + 
        sqrt(((F01 + F10)**2 + (F00 - F11)**2)*((F01 - F10)**2 + (F00 + F11)**2))) + 
      (-F00**2 + F01**2 - F10**2 + F11**2 + sqrt(((F01 + F10)**2 + (F00 - F11)**2)*
          ((F01 - F10)**2 + (F00 + F11)**2)))*sqrt(F00**2 + F01**2 + F10**2 + F11**2 + 
         sqrt(((F01 + F10)**2 + (F00 - F11)**2)*((F01 - F10)**2 + (F00 + F11)**2))))/(
     (2*sqrt(2)*sqrt(-4*(F01*F10 - F00*F11)**2 + (F00**2 + F01**2 + F10**2 + F11**2)**2)))
     
    eigU = Eig2x2(U00, U01, U01, U11)
    LE0, LE1 = [log(u) for u in eigU]
    
    if rtnU != True:
        return ( 2/3 * ( LE0**2 + LE1**2 + (-LE0-LE1)**2 ) )**0.5
    else:
        return ( ( 2/3 * ( LE0**2 + LE1**2 + (-LE0-LE1)**2 ) )**0.5,
                U00, U01, U11
                ) 

expt = argv[1]
proj = 'TTGM-{}_FS19SS6'.format(expt)
print('\n')
print(proj)

expt = int( proj.split('_')[0].split('-')[1])
FS = int( proj.split('_')[1].split('SS')[0].split('S')[1] )
SS = int( proj.split('_')[1].split('SS')[1] )
arampath = '../{}/AramisBinary'.format(proj)
prefix = '{}_'.format(proj)
savepath = '../{}/IncrementalAnalysis'.format(proj)

key = n.genfromtxt('../../ExptSummary.dat', delimiter=',')
alpha = key[ key[:,0] == int(expt), 3 ]
th = key[ key[:,0] == int(expt), 6 ][0]

# [0]Stg [1]Time [2]AxSts [3]ShSts [4]AxForce [5]Torque [6]MTS Disp [7]MTS Rot
STF = n.genfromtxt('../../{}/STF.dat'.format(proj), delimiter=',')
LL = n.genfromtxt('../../{}/prof_stages.dat'.format(proj), delimiter=',', dtype=int)[2]
last = int(STF[-1,0])

# Averaging F  for passing points
deeq2 = n.empty((last+1,5))
totstn = n.empty(last+1)

A = n.load('../{}/NewFilterPassingPoints_3med.npy'.format(savepath))[-1]
loc = A[:,15].argmax()
maxI, maxJ = A[loc,[0,1]].ravel()

# ID the site B zone.  Centered 2.5 wall thicknesses below max point on the test-section
# One wall thickness around in all direction
A = n.load('../../{}/AramisBinary/{}_{}.npy'.format(proj, proj, 0))
x, y = A[ (A[:,0] == maxI) & (A[:,1] == maxJ), 2:4 ].ravel()/th
if expt not in [7, '7']:
    xmin, xmax, ymin, ymax = x-0.5, x+0.5, y-3, y-2
else:    
    xmin, xmax, ymin, ymax = x-0.5, x+0.5, y+2, y+3
xmin, xmax, ymin, ymax = map(lambda j: j*th, (xmin, xmax, ymin, ymax))

for k in trange(1,last+1):
    
    #if k%25 == 0:  print(k, end=',', flush=True)

    sig00 = STF[k,2]/2  # Hoop sts (assumed 1/2 axial)
    sig11 = STF[k,2]   # Ax sts
    sig01 = STF[k,3]   # Sh sts
    # Mises equivalent sts
    sigvm = n.sqrt(sig11**2 + sig00**2 - sig00*sig11 + 3*sig01**2)
    # Principle stresses
    s1 = sig00/2 + sig11/2 + sqrt(sig00**2 - 2*sig00*sig11 + 4*sig01**2 + sig11**2)/2
    s2 = sig00/2 + sig11/2 - sqrt(sig00**2 - 2*sig00*sig11 + 4*sig01**2 + sig11**2)/2
    # Hosford eq. sts
    sigh8 = (((s1-s2)**8+(s2-0)**8+(0-s1)**8)/2)**(1/8)
    
    A = n.load('../../{}/AramisBinary/{}_{}.npy'.format(proj, proj, k))
    rng = (A[:,2]>=xmin) & (A[:,2]<=xmax) & (A[:,3]>=ymin) & (A[:,3]<=ymax)
    A = A.compress(rng, axis=0).take([0,1,8,9,10,11], axis=1)
        
    ##  deeq2: Average F for all passing!
    if k == 1:
        B00t, B11t = 1, 1
        B01t, B10t = 0, 0
    A00t, A01t, A10t, A11t = [i.mean() for i in A[:,2:].T]
    de00, de01, de11 = increm_strains(A00t,A01t,A10t,A11t,B00t,B01t,B10t,B11t)
    deeq2[k,0] = (sig00*de00 + sig11*de11 - 2*sig01*de01)/sigvm    
    deeq2[k,1] = (sig00*de00 + sig11*de11 - 2*sig01*de01)/sigh8        
    deeq2[k,2:5] = n.c_[de00, de01, de11]
    
    totstn[k] = LEp(A00t, A01t, A10t, A11t)

    # For next stage, keep A and assign it to B
    B00t, B01t, B10t, B11t = A00t, A01t, A10t, A11t
    
deeq2[0] = 0

for i in [deeq2]:
    i[ n.any(n.isnan(i), axis=1) ] = 0

    
headerline = ('SiteB coords Xmin, Xmax, Ymin, Ymax\n' + 
              '{:g}, {:g}, {:g}, {:g}\n'.format(xmin,xmax,ymin,ymax) + 
              '[0]eeq-VM, [1]eeq=H8, [2]e00, [3]e01, [4]e11, [5]TraditionalEeq' )
deeq2 = deeq2.cumsum(axis=0)
fname='SiteB_{}.dat'.format(proj)
n.savetxt(fname, X=n.c_[deeq2, totstn], header=headerline,
        fmt = '%.6f', delimiter=', ')

fig, ax1, ax2 = f.make21()
# [0-5]Mean VM-H8-Anis-de00-01-00, [6-11]Max VM-H8-Anis-de00-01-00, [12-13]Mean, max Classic LEp
e = n.genfromtxt('../{}/NewFilterResults_3med.dat'.format(savepath), delimiter=',')
phi = n.genfromtxt('../{}/../disp-rot.dat'.format(savepath), delimiter=',')[:,5]
ax1.plot(phi, e[:,0], label='Mean')
ax1.plot(phi, e[:,6], label='Max')
ax1.plot(phi, deeq2[:,0], label='Loc B')
ax1.axvline(phi[LL], color='k', linestyle='--')
ax1.axis(xmin=0, ymin=0)
f.eztext(ax1, 'Incremental', 'br')

ax2.plot(phi, e[:,12], label='Mean')
ax2.plot(phi, e[:,13], label='Max')
ax2.plot(phi, totstn, label='Loc B')
ax2.axvline(phi[LL], color='k', linestyle='--')
ax2.axis(xmin=0, ymin=0)
f.eztext(ax2, 'Traditional', 'br')

p.savefig('SiteB_{}.png'.format(proj), dpi=125)
