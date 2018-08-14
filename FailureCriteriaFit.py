import numpy as n
pi = n.pi
import matplotlib.pyplot as p
p.style.use('mysty')
import figfun as f
import scipy.optimize as so
from scipy.interpolate import interp1d
from pandas import read_excel
from sys import argv

try:
    constit = argv[1]
except IndexError:
    # Specify the constit you wanna use
    constit = 'VM'

# First load up the failure data 
d = read_excel('../Figs/Kaleidagraph/TTGM_SetData.xlsx', sheetname='Table 2', index_col='Expt')
d.drop([11,12,13,14,15], inplace=True)
d1 = read_excel('../Figs/Kaleidagraph/TTGM_SetData.xlsx', sheetname='Table 1', index_col='Expt')
d1.drop([11,12,13,14,15], inplace=True)
d['t_o'] = d1.t
# Convert ksi\n(MPa) strings of stress into usable floats
d.SigF = [float(d.SigF.values[j].split('\r')[0].split('\n')[0]) for j in range(len(d))]
d.TauF = [float(d.TauF.values[j].split('\r')[0].split('\n')[0]) for j in range(len(d))]
d.t_o = [float(d.t_o.values[j].split('\r')[0].split('\n')[0]) for j in range(len(d))]
Σ, efmn, efmx = d.Triax.values, d.efmn.values, d.efmx.values

if constit == 'H8':
    for i,k in enumerate(d.index):
        # [0-5]Mean VM-H8-Anis-de00-01-00, [6-11]Max VM-H8-Anis-de00-01-00, [12-13]Mean, max Classic LEp
        e = n.genfromtxt('../../TTGM-{}_FS19SS6/IncrementalAnalysis/NewFilterResults_3med.dat'.format(k), delimiter=',')[-1]
        efmn[i], efmx[i] = e[[1,7]]

efm = efmn.copy()

# Load up the constitutive model
mat = read_excel('FailureCriteriaFit_StsStnCurves.xlsx', sheet_name=constit, header=None).values

# Defining functions for the HC model

def f1(q):
    return (2/3)*n.cos(pi*(1-q)/6)
def f2(q):
    return (2/3)*n.cos(pi*(3+q)/6)
def f3(q):
    return -(2/3)*n.cos(pi*(1+q)/6)
def F(θ,a):
    return ( (1/2)*( n.abs(f1(θ) - f2(θ))**a + n.abs(f2(θ) - f3(θ))**a + n.abs(f3(θ) - f1(θ))**a ))**(1/a)
def triax(sig, tau):
    return (sig/2)/n.sqrt(.75*(sig**2+4*tau**2))

def HC(params, sig, tau, constit='VM'):
    σc, a, c = params
    η = triax(sig,tau)
    θ = -2*n.arccos(27*sig*tau**2/(4*(3*sig**2/4 + 3*tau**2)**(3/2)))/pi + 1
    rtn = σc / (F(θ,a) + c*(f1(θ) + f3(θ) + 2*η))
    if constit == 'H8':
        return rtn * F(θ,8.0)
    else:
        return rtn

def plot_HC(params, α, constit='VM'):
    σc, a, c = params
    η = α/n.sqrt(12+3*α**2)
    θ = 1-(2/pi)*n.arccos( 27*.25*α / (.75*α**2 + 3)**1.5)
    rtn = σc / (F(θ,a) + c*(f1(θ) + f3(θ) + 2*η))
    if constit == 'H8':
        return rtn * F(θ,8.0)
    else:
        return rtn    
    
def err(params, sig, tau, sigeq_fail, constit='VM'):
        return HC(params, sig, tau, constit) - sigeq_fail
    
bounds = ((0, 0, 0),(n.inf, 2, n.inf))

# Calculate the tru thickness at failure in each expt
d['t'] = n.ones(len(d))
for k in d.index:
    # [0-5]Mean VM-H8-Anis-de00-01-00, [6-11]Max VM-H8-Anis-de00-01-00, [12-13]Mean, max Classic LEp
    e = n.genfromtxt('../../TTGM-{}_FS19SS6/IncrementalAnalysis/NewFilterResults_3med.dat'.format(k), delimiter=',')[-1]
    e = n.array([[e[3], e[4]],[e[4],e[5]]])
    e1, e2 = n.linalg.eigvalsh(e)
    d.loc[k].t = d.loc[k].t_o * n.exp(-(e1+e2))

# Now calc. the tru sts at failure
d.SigF *= d.t_o / d.t
d.TauF *= d.t_o / d.t

# And the VM Eq.sts
d['VM'] = n.sqrt( ( (d.SigF/2 - d.SigF)**2 + (d.SigF)**2 + (-d.SigF/2)**2 + 6*d.TauF**2)/2 )
sig, tau, se = [d[j].values for j in ('SigF', 'TauF','VM')]

# A nice way to check that I am calculating d.VM and my f1, f2 f3 functions correctly!
σ1 = d.VM * (f1(d['Q-bar']) + d.Triax)
σ2 = d.VM * (f2(d['Q-bar']) + d.Triax)
σ3 = d.VM * (f3(d['Q-bar']) + d.Triax)
assert n.allclose( (((1/2)*((σ1-σ2)**2 + (σ2-σ3)**2 +(σ3-σ1)**2))**.5).values, d.VM.values)

# And finally the H8 Eq sts
d['H8'] = (((1/2)*((σ1-σ2)**8 + (σ2-σ3)**8 +(σ3-σ1)**8))**.125)

# Optimizing to the stress I get if I interpolate from the sts-stn curve using the failure strain!
# Optimizing to the true stress I calculated at failure

sts = interp1d(mat[:,0], mat[:,1], fill_value='extrapolate').__call__(efm)

if constit == 'H8':
    res = so.least_squares(err, x0=[1,1,1], args=(d.SigF.values, d.TauF.values, sts, constit), bounds=bounds)
else:
    res = so.least_squares(err, x0=[1,1,1], args=(d.SigF.values, d.TauF.values, sts, constit), bounds=bounds)

print(res.x)

fig, ax1, ax2 = f.make12()

α = n.empty(1000)
α[:-50] = n.linspace(0.0,8,950)
α[-50:] = n.linspace(8,20,50)
η = α/n.sqrt(12+3*α**2)


ax1.plot(η, plot_HC(res.x, α, constit), label=('{}\n$\\sigma_c$,a,c\n' + '{:.3f}\n'*3).format(constit,*res.x))
ax1.plot(d.Triax, sts, 'o')
ax1.set_xlabel('$\\sigma_m/\\sigma_e$')
ax1.set_ylabel('$\\sigma_e^f$')
f.ezlegend(ax1)

eef_pred = interp1d(mat[:,1], mat[:,0], fill_value='extrapolate').__call__(plot_HC(res.x, α, constit))
ax2.plot(η, eef_pred)
ax2.axis([0,.6,0,2])
ax2.set_xlabel('$\\sigma_m/\\sigma_e$')
ax2.set_ylabel('$e_e^f$')
ax2.plot(d.Triax, d.efmn, 'o')

[f.myax(x) for x in (ax1,ax2)]

fig.savefig('Failure_HC_{}'.format(constit), dpi=125)
    
# First load up the failure data 
p.figure()
def JC(triax, D1, D2, D3):
    return D1 + D2*n.exp(D3*triax)

bounds = ((0,-n.inf, -n.inf), (n.inf, n.inf, 0))

for efm, lab in zip([efmx, efmn], ['Max','Mean']):
    params, info = so.curve_fit(JC, Σ, efm, bounds=bounds)
    x = n.linspace(0, 2/3)
    p.plot(x, JC(x,*params), label='{}\nD1={:.3f}\nD2={:.3f}\nD3={:.3f}'.format(lab, *params))
    p.plot(Σ, efm, '.')

f.ezlegend(p.gca())
f.eztext(p.gca(), '{}\nJohnson-Cook'.format(constit), 'ur')
p.xlabel('$\\sigma_m/\\sigma_e$')
p.ylabel('$e_e^f$')
p.axis(xmax=0.6)
f.myax(p.gca())

p.savefig('Failure_JC_{}'.format(constit), dpi=125)    