import numpy as n
import matplotlib.pyplot as p
from numpy import sqrt, arccos, pi, array
import figfun as f

beta = {0.5:0.60, 0.75:0.47, 1.0:0.39,
        1.5:0.32, 2.00:0.25, 3.0:0.23,
        4.0:0.21, n.nan:0.19}

def stress_measures(sig, tau, alpha):
    if n.isnan(alpha):
        alpha = n.nan
    hoop = beta[alpha]*sig
    #hoop = 0.5*sig
    sm = (sig+hoop)/3
    se = (.5*((sig-hoop)**2 + hoop**2 + sig**2 + 6*tau**2))**0.5
    triax_beta = sm/se
    triax = (sig/2)/sqrt(.75*(sig**2+4*tau**2))
    qbar = -2*arccos(27*sig*tau**2/(4*(3*sig**2/4 + 3*tau**2)**(3/2)))/pi + 1
    Sp = array([3*sig/4 - sqrt(sig**2 + 16*tau**2)/4, 
                    3*sig/4 + sqrt(sig**2 + 16*tau**2)/4, 0])
    mu = (2*n.median(Sp) - Sp.max() - Sp.min())/(Sp.max()-Sp.min())
    return triax, triax_beta

ex = n.genfromtxt('../../ExptSummary.dat', delimiter=',')
ex = ex[ (ex[:,1] == 0) & (ex[:,3]!= 0) ]
p.style.use('mysty')
fig, ax = p.subplots()
for k,(x,alp) in enumerate(zip(ex[:,0].astype(int), ex[:,3])):
    lines = []
    proj = 'TTGM-{}_FS19SS6'.format(x)
    # [0-5]Mean VM-H8-Anis-de00-01-00, [6-11]Max VM-H8-Anis-de00-01-00, [12-13]Mean, max Classic LEp
    e = n.genfromtxt('../../{}/IncrementalAnalysis/NewFilterResults_3med.dat'.format(proj), delimiter=',')[-1]
    sig, tau = n.genfromtxt('../../{}/STF.dat'.format(proj), delimiter=',', unpack=True, usecols=(2,3))
    triax, triax2 = stress_measures(sig[-1], tau[-1], alp)
    if x == ex[-1,0]:
        ax.plot(triax, e[0], 'C0s')#, label='$\\beta=\\mathsf{1}$/$\\mathsf{2}$')
        ax.plot(triax2, e[0], 'C1o')#, label='$\\beta=\\mathsf{f}(\\alpha)$')
    else:
        ax.plot(triax, e[0], 'C0s')
        ax.plot(triax2, e[0], 'C1o')

#f.ezlegend(ax,markers=True,fontsize=20, loc=3)
f.eztext(ax, '$\\beta=\\mathsf{1}$/$\\mathsf{2}$\n', 'bl', color='C0')
f.eztext(ax, '$\\beta=\\mathsf{f}(\\alpha)$', 'bl', color='C1')
ax.axis(xmin=0,ymin=0,ymax=1.65)
ax.set_xlabel('$\\sigma_{\\mathsf{m}}/\\sigma_{\\mathsf{e}}$')
ax.set_ylabel('$\\mathsf{e}^{\\mathsf{p}}_{\\mathsf{e}}$')
f.eztext(ax, 'Al-6061-T6\nMean values', 'ur')
f.myax(ax)
p.savefig('Triax_Beta.pdf')
p.show()

