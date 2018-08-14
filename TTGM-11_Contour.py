import numpy as n
from sys import argv
from numpy.linalg import eigvalsh
import matplotlib.pyplot as p
import matplotlib.tri as tri
p.style.use('mysty')
import figfun as f

save = 0
exp = [ 11]
alpha = [ 0.0]
FS,SS = 60,8
try:
    cmap = argv[1]
except IndexError:
    cmap='viridis'
e01 = True
min2zero = True

def mysqrtm(X):
    from numpy import sqrt, hstack, vstack, dstack
    import numexpr as ne
    ### 0,0
    A = X[:,0,0]
    B = X[:,0,1]
    C = X[:,1,0]
    D = X[:,1,1]
    U00 = ne.evaluate("( (sqrt(A + D - sqrt(A**2 + 4*B*C - 2*A*D + D**2))*(-A + D + sqrt(A**2 + 4*B*C - 2*A*D + D**2)))-((-A + D - sqrt(A**2 + 4*B*C - 2*A*D + D**2))*sqrt(A + D + sqrt(A**2 + 4*B*C - 2*A*D + D**2))) )/(2*sqrt(2)*sqrt(A**2 + 4*B*C - 2*A*D + D**2))")

    ### 0,1
    U01 = ne.evaluate("(B*(-sqrt(A - sqrt(4*B*C + (A - D)**2) + D) + sqrt(A + sqrt(4*B*C + (A - D)**2) + D)))/(sqrt(2)*sqrt(4*B*C + (A - D)**2))")
        
    ### 1,0
    U10 = ne.evaluate("(C*(-sqrt(A - sqrt(4*B*C + (A - D)**2) + D) + sqrt(A + sqrt(4*B*C + (A - D)**2) + D)))/(sqrt(2)*sqrt(4*B*C + (A - D)**2))")
     
    ### 1,1    
    U11 = ne.evaluate("( (A + sqrt(4*B*C + (A - D)**2) - D)*sqrt(A - sqrt(4*B*C + (A - D)**2) + D) + (-A + sqrt(4*B*C + (A - D)**2) + D)*sqrt(A + sqrt(4*B*C + (A - D)**2) + D) )/(2*sqrt(2)*sqrt(4*B*C + (A - D)**2))")

    return hstack((dstack( (U00[:,None][:,:,None],U01[:,None][:,:,None]) ),dstack((U10[:,None][:,:,None],U11[:,None][:,:,None]))))

folder = 'TTGM-{}_FS{}SS{}'.format(exp[0],FS,SS)
stf = n.genfromtxt('../../{}/STF.dat'.format(folder), delimiter=',', dtype=int)
last = stf[-1,0]
max10 = n.genfromtxt('../../{}/Max10.dat'.format(folder), delimiter=',')
maxval = max10[0,0]
Xmin,Xmax,Ymin,Ymax = n.genfromtxt('../../{}/box_limits.dat'.format(folder), delimiter=',')
box_path = n.array([Xmin,Ymin,Xmin,Ymax,Xmax,Ymax,Xmax,Ymin,Xmin,Ymin]).reshape(-1,2)

if e01:
    A = n.load('../../{}/IncrementalAnalysis/PointsInLastWithStrains.npy'.format(folder))[-1]
    colorval = -A[:,13]
    clabel = 'e$_{\\theta\\mathsf{x}}$'
else:
    A = n.load('../../{}/AramisBinary/{}_{}.npy'.format(folder,folder,last))
    clabel = 'e$_\\mathsf{e}$'
    F=A[:,-4:].reshape(len(A[:,0]),2,2) 
    FtF = n.einsum('...ji,...jk',F,F) #Same as that commented out above. Kept the comment to recall how to transpose a stack of matrices
    U = mysqrtm( FtF )
    eigU = eigvalsh(U) #dimension is len(A[:,0]) x 2.  Each row is the vector of eigenvalues for that row's matrix
    LE = n.log(eigU)
    LE0,LE1 = LE[:,0], LE[:,1]
    LEp = ( 2/3 * ( LE0**2 + LE1**2 + (-LE0-LE1)**2 ) )**0.5
    colorval = LEp

rng = ((A[:,3] >= -0.2) & (A[:,3]<=0.2))    
A = A[rng]
colorval = colorval[rng]
x=A[:,2].copy()
y=A[:,3].copy()
x/=(1.72/2)
y/=.0461    
    
maxval = n.sort(colorval)[-500:].mean()
if min2zero:
    minval = 0
else:
    minval = n.sort(colorval)[:500].mean()
crange = n.linspace(minval, maxval, 100)
crange = n.linspace(0,1,100)

W,H = 12,6
fig = p.figure(figsize=(W,H))
xo,yo,w,h = 1,1,8,4
mult = [1/W,1/H,1/W,1/H]
ax = fig.add_axes(n.array([xo,yo,w,h])*mult)

# For some reason, after normalizing x and y by R and t, the triangulation changes
# and as a result the tricontour looks strange
# So I;m going thu the hassle of computing the triangulation using the non-normalized coords
Z = tri.Triangulation(A[:,2],A[:,3])
#tric = p.tricontourf(x,y,LEp,crange,extend='both',cmap=cmap)

tric = p.tricontourf(x,y,Z.triangles,colorval,crange,extend='both',cmap=cmap)

ax.axis([n.min(x)*1.1, 1.1*n.max(x), -.2/.0461, .2/.0461])
#ax.plot(box_path[:,0], box_path[:,1], 'w')
#p.axis('equal')
ax.set_xlabel('$\\mathsf{x}_\\mathsf{o}/\\mathsf{R}_\\mathsf{o}$')
ax.set_ylabel('$\\frac{\\mathsf{y}_\\mathsf{o}}{\\mathsf{t}_\\mathsf{o}}$')    
f.myax(ax)
ax_bar = fig.add_axes(n.array([9.5,1,.25,4])*mult)
ticklocs = n.abs(n.unique(crange//.1*.1))
ticklabs = ['{:g}'.format(L) for L in ticklocs]
cbar = p.colorbar(tric, ax_bar, extendrect=True, extendfrac=0, 
                    ticks=ticklocs, format='%.1f')

cbar.set_label(clabel)
f.colorbar(ax,cbar)
if save:
    p.savefig('TTGM-11.png'.format(cmap), bbox_inches='tight', dpi=400)
    # This gets rid of the weird aliasing effect for the PDF, but introduces some anomalous things into the png
    for c in tric.collections:
        c.set_edgecolor('face')
    p.savefig('TTGM-11.pdf'.format(cmap), bbox_inches='tight')
p.show()