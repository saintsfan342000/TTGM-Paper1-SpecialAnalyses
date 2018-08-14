import sympy as sp
sp.init_printing()
from sympy.utilities.autowrap import ufuncify, autowrap
import scipy.optimize as so
import numpy as n
import matplotlib.pyplot as p

sp.var('sx,sq,sqx,a,b')
wrappers = [0,0,0]
als = n.arange(0.01,6.02,0.02)
als[-1] = 10000
betas = n.empty((len(als),len(wrappers)))
labels = ['vm','h8','anis']

kelin = {0.5:0.60, 0.75:0.47, 1.0:0.39,
        1.5:0.32, 2.00:0.25, 3.0:0.23,
        4.0:0.21, n.nan:0.19}

for z,name in enumerate(labels):

    if name == 'anis':
        (cp12,cp13,cp21,cp23,
         cp31,cp32,cp44,cp55,cp66) = (0.6787,0.98478,1.1496,1.0275,
                                      0.94143, 1.1621,1.3672,1,1)
        (cpp12,cpp13,cpp21,cpp23,
         cpp31,cpp32,cpp44,cpp55,cpp66) = (1.0563,0.96185,0.68305,0.71287,
                                       1.0933,0.84744,0.69489,1,1)
    else:
        (cp12,cp13,cp21,cp23,
             cp31,cp32,cp44,cp55,cp66) = [1 for i in range(9)]
        (cpp12,cpp13,cpp21,cpp23,
            cpp31,cpp32,cpp44,cpp55,cpp66) = [1 for i in range(9)]

    Cp = sp.zeros(6,6)
    Cp[0,1], Cp[0,2] = -cp12, -cp13
    Cp[1,0], Cp[1,2] = -cp21, -cp23
    Cp[2,0], Cp[2,1] = -cp31, -cp32
    Cp[3,3], Cp[4,4], Cp[5,5] = cp44, cp55, cp66

    Cpp = sp.zeros(6,6)
    Cpp[0,1], Cpp[0,2] = -cpp12, -cpp13
    Cpp[1,0], Cpp[1,2] = -cpp21, -cpp23
    Cpp[2,0], Cpp[2,1] = -cpp31, -cpp32
    Cpp[3,3], Cpp[4,4], Cpp[5,5] = cpp44, cpp55, cpp66

    T = sp.zeros(6,6)
    T[0,0], T[0,1], T[0,2] = 2, -1, -1
    T[1,0], T[1,1], T[1,2] = -1, 2, -1
    T[2,0], T[2,1], T[2,2] = -1, -1, 2
    T[3,3], T[4,4], T[5,5] = 3, 3, 3
    T*=sp.Rational(1,3)

    s = sp.Matrix([0,sq,sx,0,0,sqx])

    def vectomat(x):
        return sp.Matrix([[x[0], x[3], x[4]], [x[3], x[1], x[5]],[ x[4], x[5], x[2]]])

    Sp = vectomat(Cp*T*s)
    Spp = vectomat(Cpp*T*s)

    Sp = list(Sp.eigenvals())
    Spp = list(Spp.eigenvals())

    if name in ['h8', 'anis']:
        k = 8.0
    else:
        k = 2.0

    PHI = ( (Sp[0]-Spp[0])**k + 
          (Sp[0]-Spp[1])**k + 
          (Sp[0]-Spp[2])**k + 
          (Sp[1]-Spp[0])**k + 
          (Sp[1]-Spp[1])**k + 
          (Sp[1]-Spp[2])**k + 
          (Sp[2]-Spp[0])**k + 
          (Sp[2]-Spp[1])**k + 
          (Sp[2]-Spp[2])**k
        )

    dfdq = PHI.diff(sq)
    #dfdq = dfdq.subs(((sqx,sx/a),(sx,sq/b),(sq,1)))
    dfdq = dfdq.subs(sq,b*sx)
    dfdq = dfdq.subs(sx,a*sqx)
    dfdq = dfdq.subs(sqx,1.0)
    dfdq = sp.N(dfdq)

    try:
        wrapped = autowrap(dfdq, args=(a,b))
    except:
        wrapped = sp.lambdify((a,b), dfdq)
    wrappers[z] = wrapped

    def fun(b,a):
        return wrapped(a,b)

    for j,i in enumerate(als):
        #betas[j,z] = so.newton(fun,1,args=(i,))
        betas[j,z] = so.brentq(fun,0.3,1,args=(i,))


import figfun as f
p.style.use('mysty')
fig, ax = p.subplots()
for i in range(z+1):
    label=labels[i].upper()+'\n$\\beta_\\infty$={:.2f}'.format(betas[-1,i])
    p.plot(als[:-1], betas[:-1,i], label=label)

p.xlabel('$\\alpha$')
p.ylabel('$\\beta$')
f.ezlegend(ax)
f.eztext(ax, 'Al 6061-T6\n$\\mathsf{d}\\epsilon_\\mathsf{11}=\\mathsf{0}$', 'ur')
f.myax(ax)
p.savefig('PlaneStn_YldFun.pdf')
p.show()

n.savetxt('PlaneStn_YldFun.dat', fmt='%.6f', delimiter=',', X=n.c_[als,betas], header='[0]Alpha, [1]VM, [2]H8, [3]Anis')
