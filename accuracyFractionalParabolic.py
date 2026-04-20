from fenics import *
import sympy2fenics as sf
import math

def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))

# ******* Exact solutions and forcing terms for error analysis ****** #
Tfinal = 1.0; dt_initial = 0.01; 

alpha = 0.5  # Fractional order

u_str = 't^(2+alpha) * cos(pi*x) * sin(pi*y)'

Gamma_ratio = math.gamma(3 + alpha) / math.gamma(3) # This ratio arises from the Caputo derivative of t^(2+alpha) being Gamma(3+alpha)/Gamma(3)*t^2
Dt_alpha_u_str = 'Gamma_ratio * pow(t,2) * cos(pi*x) * sin(pi*y)'



K = as_tensor(((2,0),(0,0.1)))

nkmax = 4
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

    

    # ********* Finite dimensional spaces ********* #
    Vh = FunctionSpace(mesh, 'CG', 1)
    W2h = FunctionSpace(mesh_coarse, 'CG', 2)
    nn.append(Vh.dim())
    
    # ********* test and trial functions ****** #
    v = TestFunction(Vh)
    u = TrialFunction(Vh)
    u_h = Function(Vh)
    # ********* instantiation of exact solutions ****** #
    t = 0.0; dt = dt_initial

    u_ex = Expression(str2exp(u_str), t=t, alpha = alpha, degree=6, domain=mesh)
    Dt_alpha_u_ex = Expression(str2exp(Dt_alpha_u_str), t=t, Gamma_ratio=Gamma_ratio, alpha=alpha, degree=6, domain=mesh)
    u_ex_coarse = Expression(str2exp(u_str), t=t, alpha=alpha, degree=6, domain=mesh_coarse)

    u_old = interpolate(u_ex, Vh) # initial condition at t=0

    u_history = [Function(Vh) for _ in range(1000)]  # Pre-allocate history of solutions for fractional derivative approximation
    u_history[0].assign(u_old) 

    # Weights calculation: a_j = (j+1)^(1-alpha) - j^(1-alpha)
    def get_weight(j, alpha):
        return pow(j + 1, 1 - alpha) - pow(j, 1 - alpha)

    # Note: In the L1 scheme, the "current" unknown is u_h (at t_n)
    # The term (u^n - u^{n-1}) is handled separately from the summation of previous steps.
    
    E_u = 0; E_Iu = 0
    step = 1

    # **** time-stepping loop **** #
    while t < Tfinal - 1e-12:
        if t + dt > Tfinal:
            dt = Tfinal - t 
        t += dt
        print("    Time: t = {:.4f}".format(t))
        C_alpha = pow(dt, -alpha) / math.gamma(2 - alpha)

        u_ex.t = t
        Dt_alpha_u_ex.t = t
        u_ex_coarse.t = t 
        f_ex = Dt_alpha_u_ex - div(K*grad(u_ex))

        history_sum = Function(Vh)
        history_sum.assign(Constant(0.0))

        for j in range(1, step):
            weight = get_weight(j, alpha)
            # u_history indices: u_history[n-j] is u_history[step-j]
            diff = Function(Vh)
            diff.assign(u_history[step - j])
            diff.vector().axpy(-1.0, u_history[step - j - 1].vector())
            # Add weighted difference
            history_sum.vector().axpy(weight, diff.vector())

        # ********* Weak forms ********* #
        auv = C_alpha*u*v*dx +dot(K*grad(u), grad(v))*dx
        Fv = (f_ex + C_alpha * (u_history[step-1] - history_sum)) * v * dx

        
        # ********* boundary conditions (Essential) ******** #
        bcU = DirichletBC(Vh, u_ex, 'on_boundary')
        u_curr = Function(Vh)
        # ********* solve for u_h at the new time step
        solve(auv == Fv, u_curr, bcU)
        # Update for next time step
        u_history[step] = Function(Vh)
        u_history[step].assign(u_curr)
        u_h.assign(u_curr)
        step += 1

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