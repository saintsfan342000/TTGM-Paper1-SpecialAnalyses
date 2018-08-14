import numpy as n
pi = n.pi
import matplotlib.pyplot as p
p.style.use('mysty-sub')
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

x = [0 ,0]

# First load up the failure data 
x[1] = read_excel('../Figs/Kaleidagraph/TTGM_SetData.xlsx', sheetname='FailureStrain', index_col='Expt')

x[0] = read_excel('../../../../AAA_TensionTorsion/TT2_SetData.xlsx', sheetname='FailureStrain', index_col='Expt')
x[0].drop(17, inplace=True)

fig, ax1, ax2 = f.make21()

labs = ['GM', 'TT2'][::-1]
marks = 'o','^'
for k,d in enumerate(x):
    ax1.plot(d.Triax.values, d['VM-Mean'].values, marks[k], label=labs[k])
    ax2.plot(d.Triax.values, d['VM-Max'].values, marks[k], label=labs[k])

ax1.set_ylabel('$\\bar{e}_e^f$')
ax2.set_ylabel('$e_e^f$')
f.eztext(ax1, 'Mean', 'ur')
f.eztext(ax2, 'Max', 'ur')

for ax in [ax1, ax2]:
    f.eztext(ax, '3D DIC\nAl 6061-T6', 'll')
    f.ezlegend(ax, markers=True)
    ax.set_xlabel('$\\sigma_m/\\sigma_e$')
    ax.axis([0,0.6,0,1.8])
    f.myax(ax)

p.savefig('CompareGMTT2.png', dpi=125)
p.show()
