from fenics import *

import sympy2fenics as sf
def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))

# ******* Exact solutions and forcing terms for error analysis ****** #

u_str = 'cos(pi*x)*sin(pi*y)'

K = as_tensor(((2,0),(0,0.1)))

r=1; nkmax = 4

hh = []; nn = []; eu = []; ru = [];
e0 = []; r0 = []

ru.append(0.0); r0.append(0.0); 


for nk in range(nkmax):
    print("....... Refinement level : nk = ", nk)
    
    nps = pow(2,nk+1)
    mesh = UnitSquareMesh(nps,nps)
    hh.append(mesh.hmax())

    # ********* Finite dimensional spaces ********* #
    Vh = FunctionSpace(mesh, 'CG', r) # Lagrange = CG (continuous Galerkin)
    
    nn.append(Vh.dim())
    
    # ********* test and trial functions ****** #
    v = TestFunction(Vh)
    u = TrialFunction(Vh)
    
    # ********* instantiation of exact solutions ****** #
    
    u_ex    = Expression(str2exp(u_str), degree=6, domain=mesh)
    f_ex    = - div(K*grad(u_ex))

    # ********* boundary conditions (Essential) ******** #
    
    bcU = DirichletBC(Vh, u_ex, 'on_boundary')
    
    # ********* Weak forms ********* #
    
    auv = dot(K*grad(u),grad(v))*dx  #    
    Fv  = f_ex*v*dx 

    u_h = Function(Vh)
    
    solve(auv == Fv, u_h, bcU) #using a direct solver
    
    # ********* Computing errors ****** #

    E_u_H1 = assemble((u_ex-u_h)**2*dx \
                      +(grad(u_ex)-grad(u_h))**2*dx)
    E_u_L2 = assemble((u_ex-u_h)**2*dx)

    eu.append(pow(E_u_H1,0.5))
    e0.append(pow(E_u_L2,0.5))

    # or simply eu.append(errornorm(u_ex,u_h,'L2'))
    
    if(nk>0):
        ru.append(ln(eu[nk]/eu[nk-1])/ln(hh[nk]/hh[nk-1]))
        r0.append(ln(e0[nk]/e0[nk-1])/ln(hh[nk]/hh[nk-1]))
        

# ********* Generating error history ****** #
print('====================================================')
print('  DoF      h    e_1(u)   r_1(u)   e_0(u)  r_0(u)    ')
print('====================================================')
for nk in range(nkmax):
    print('{:6d}  {:.4f} {:6.2e}  {:.3f}  {:6.2e}  {:.3f} '.format(nn[nk], hh[nk], eu[nk], ru[nk], e0[nk], r0[nk]))
print('====================================================')
