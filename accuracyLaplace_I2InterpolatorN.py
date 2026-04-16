from fenics import *
import numpy as np
from scipy.spatial import KDTree
import sympy2fenics as sf

def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))

# ******* Exact solutions ****** #
u_str = 'cos(pi*x)*sin(pi*y)'
K = as_tensor(((2,0),(0,0.1)))

nkmax = 4
hh = []; nn = []; eu = []; ru = []
e0 = []; r0 = []; eIu1 = []; eIu0 = []
rIu1 = []; rIu0 = []

ru.append(0.0); r0.append(0.0); rIu1.append(0.0); rIu0.append(0.0)

for nk in range(nkmax):
    print("....... Refinement level : nk = ", nk)
    
    nps = pow(2, nk+1)
    mesh = UnitSquareMesh(nps, nps)
    mesh_coarse = UnitSquareMesh(pow(2, nk), pow(2, nk))
    hh.append(mesh.hmax())

    # ********* Finite dimensional spaces ********* #
    Vh = FunctionSpace(mesh, 'CG', 1)
    W2h = FunctionSpace(mesh_coarse, 'CG', 2)
    nn.append(Vh.dim())
    
    # ********* Solve for u_h ****** #
    v = TestFunction(Vh)
    u = TrialFunction(Vh)
    
    u_ex = Expression(str2exp(u_str), degree=6, domain=mesh)
    u_ex_coarse = Expression(str2exp(u_str), degree=6, domain=mesh_coarse)
    f_ex = -div(K*grad(u_ex))
    
    bcU = DirichletBC(Vh, u_ex, 'on_boundary')
    
    auv = dot(K*grad(u), grad(v))*dx
    Fv = f_ex*v*dx
    u_h = Function(Vh)
    solve(auv == Fv, u_h, bcU)

    # ********* Use L2 projection for optimal convergence ********* #
    # This is the proper way to get O(h^3) in L2
    
    # Define the L2 projection operator from L2 to W2h
    Iu = Function(W2h)
    
    # Project u_ex onto W2h using L2 projection
    # This gives optimal convergence rates
    v_2h = TestFunction(W2h)
    u_2h = TrialFunction(W2h)
    
    a_proj = u_2h * v_2h * dx(domain=mesh_coarse)
    L_proj = u_ex_coarse * v_2h * dx(domain=mesh_coarse)
    
    solve(a_proj == L_proj, Iu)
    
    # ********* Computing errors ****** #
    E_u_H1 = assemble((u_ex - u_h)**2 * dx + dot(grad(u_ex) - grad(u_h), grad(u_ex) - grad(u_h)) * dx)
    E_u_L2 = assemble((u_ex - u_h)**2 * dx)
    
    E_Iu_H1 = assemble(dot(grad(u_ex_coarse) - grad(Iu), grad(u_ex_coarse) - grad(Iu)) * dx(domain=mesh_coarse))
    E_Iu_L2 = assemble((u_ex_coarse - Iu)**2 * dx(domain=mesh_coarse))
    
    eu.append(pow(E_u_H1, 0.5))
    e0.append(pow(E_u_L2, 0.5))
    eIu1.append(pow(E_Iu_H1, 0.5))
    eIu0.append(pow(E_Iu_L2, 0.5))
    
    if nk > 0:
        ru.append(ln(eu[nk]/eu[nk-1])/ln(hh[nk]/hh[nk-1]))
        r0.append(ln(e0[nk]/e0[nk-1])/ln(hh[nk]/hh[nk-1]))
        rIu1.append(ln(eIu1[nk]/eIu1[nk-1])/ln(hh[nk]/hh[nk-1]))
        rIu0.append(ln(eIu0[nk]/eIu0[nk-1])/ln(hh[nk]/hh[nk-1]))

# ********* Print results ****** #
print('====================================================')
print('  DoF      h    e_1(u)   r_1(u)   e_0(u)  r_0(u)  e_1(Iu)  r_1(Iu)  e_0(Iu)  r_0(Iu)    ')
print('====================================================')
for nk in range(nkmax):
    print('{:6d}  {:.4f} {:6.2e}  {:.3f}  {:6.2e}  {:.3f} {:6.2e}  {:.3f}  {:6.2e}  {:.3f} '.format(
          nn[nk], hh[nk], eu[nk], ru[nk], e0[nk], r0[nk], eIu1[nk], rIu1[nk], eIu0[nk], rIu0[nk]))
print('====================================================')