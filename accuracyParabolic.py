from fenics import *
import numpy as np
from accuracyLaplace_I2InterpolatorN import Iu
import sympy2fenics as sf

def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))

# ******* Exact solutions and forcing terms for error analysis ****** #
Tfinal = 1.0; 

u_str = 'sin(t)*cos(pi*x)*sin(pi*y)'
dt_u_str = 'cos(t)*cos(pi*x)*sin(pi*y)'

K = as_tensor(((2,0),(0,0.1)))

nkmax = 5
hh = []; nn = []; eu = []; ru = []
eIu = []; rIu = []

ru.append(0.0); rIu.append(0.0); 

# Main loop
for nk in range(nkmax):
    print("....... Refinement level : nk = ", nk)
    
    nps = pow(2, nk+1)
    mesh = UnitSquareMesh(nps, nps)
    mesh_coarse = UnitSquareMesh(pow(2, nk), pow(2, nk))
    hh.append(mesh.hmax())
    dt = 0.01 #mesh.hmax() 

    # ********* Finite dimensional spaces ********* #
    Vh = FunctionSpace(mesh, 'CG', 1)
    W2h = FunctionSpace(mesh_coarse, 'CG', 2)
    nn.append(Vh.dim())
    
    # ********* test and trial functions ****** #
    v = TestFunction(Vh)
    u = TrialFunction(Vh)
    u_h = Function(Vh)
    # ********* instantiation of exact solutions ****** #
    t = 0.0

    u_ex = Expression(str2exp(u_str), t=t, degree=6, domain=mesh)
    dt_u_ex = Expression(str2exp(dt_u_str), t=t, degree=6, domain=mesh)

    u_ex_coarse = Expression(str2exp(u_str), t=t, degree=6, domain=mesh_coarse)

    f_ex = dt_u_ex - div(K*grad(u_ex))

    u_old = interpolate(u_ex, Vh) # initial condition at t=0

    
    # ********* Weak forms ********* #
    auv = (1/dt)*u*v*dx +dot(K*grad(u), grad(v))*dx
    Fv = (1/dt)*u_old*v*dx + f_ex*v*dx

    E_u = 0; E_Iu = 0
    # **** time-stepping loop **** #
    while t < Tfinal:
        if t + dt > Tfinal:
            dt = Tfinal - t 
            t = Tfinal
        else:
            t += dt
        print("    Time step: t = {:.4f}".format(t))

        u_ex.t = t
        dt_u_ex.t = t
        u_ex_coarse.t = t
        # ********* boundary conditions (Essential) ******** #
        bcU = DirichletBC(Vh, u_ex, 'on_boundary')
        
        # ********* solve for u_h at the new time step
        solve(auv == Fv, u_h, bcU)
        # Update for next time step
        assign(u_old, u_h)

        # ********* Build the quasi-interpolant of the current solution ********* #
        # Method 1: Patch averaging with local polynomial reproduction
        #Iu = build_patch_averaged_interpolant(u_h, mesh_coarse, W2h)
        # Method 2: Standard interpolation (for comparison)
        Iu = interpolate(u_h, W2h)
 
        # either with the L2 norm in time:
        #E_u += assemble(dot(grad(u_ex) - grad(u_h), grad(u_ex) - grad(u_h)) * dx)
        #E_Iu += assemble(dot(grad(u_ex_coarse) - grad(Iu), grad(u_ex_coarse) - grad(Iu)) * dx(domain=mesh_coarse))
 
        # or with the Linfty norm in time:
        E_u = max(E_u, assemble(dot(grad(u_ex) - grad(u_h), grad(u_ex) - grad(u_h)) * dx))
        E_Iu = max(E_Iu, assemble(dot(grad(u_ex_coarse) - grad(Iu), grad(u_ex_coarse) - grad(Iu)) * dx(domain=mesh_coarse)))
    
    eu.append(pow(E_u, 0.5))
    eIu.append(pow(E_Iu, 0.5))
    
    if nk > 0:
        ru.append(ln(eu[nk]/eu[nk-1])/ln(hh[nk]/hh[nk-1]))
        rIu.append(ln(eIu[nk]/eIu[nk-1])/ln(hh[nk]/hh[nk-1]))
         

# ********* Generating error history ****** #
print('\n' + '='*50)
print('  DoF      h    e_1(u)   r_1(u)   e_1(Iu)  r_1(Iu)   ')
print('='*50)
for nk in range(nkmax):
    print('{:6d}  {:.4f} {:6.2e}  {:.3f}  {:6.2e}  {:.3f} '.format(
          nn[nk], hh[nk], eu[nk], ru[nk], eIu[nk], rIu[nk]))
print('='*50)