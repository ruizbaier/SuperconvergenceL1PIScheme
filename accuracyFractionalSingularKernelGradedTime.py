from fenics import *
import sympy2fenics as sf
import math
import numpy as np

def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))

# ******* Exact solutions and forcing terms for error analysis ****** #
Tfinal = 0.5
nsteps = 50          # total number of intervals
alpha  = 0.5         # fractional order
lambda_r = 0.3        # order of weakly singular integral (0 < lambda < 1)
gamma_coeff = 0.1    # coefficient in front of the integral term 

u_str = 't^(2+alpha) * cos(pi*x) * sin(pi*y)'
Gamma_ratio = math.gamma(3 + alpha) / math.gamma(3)
Dt_alpha_u_str = 'Gamma_ratio * pow(t,2) * cos(pi*x) * sin(pi*y)'

# Factor for the exact weakly singular integral I^lambda u_ex
integral_factor = math.gamma(3 + alpha) / math.gamma(lambda_r + 3 + alpha)
I_lambda_u_str = 'integral_factor * t^(lambda_r+2+alpha) * cos(pi*x) * sin(pi*y)'

K = as_tensor(((2, 0), (0, 0.1)))

# Grading parameter (same as used in PI scheme)
r = (2 - alpha) / alpha
time_vec = np.array([Tfinal * (i / nsteps) ** r for i in range(nsteps + 1)])
dt_vec = np.diff(time_vec)

# **** coefficients for the PI scheme for the weakly singular integral term ****
def phi_fun(n, l, xi, gamma_g):
    if l > n:
        return 0.0
    num = (n**gamma_g - (l-1)**gamma_g)**(xi+1) \
        - (n**gamma_g - l**gamma_g)**(xi+1)
    den = l**gamma_g - (l-1)**gamma_g
    return num / den

def compute_pi_coefficients(N, gamma_g, xi):
    b = np.zeros(N)
    d = np.zeros((N, N))
    for n in range(1, N+1):
        idx = n-1
        b[idx] = (n**gamma_g - 1)**(xi+1) \
               - n**(xi*gamma_g) * (n**gamma_g - xi - 1)
        for l in range(1, n+1):
            phi1 = phi_fun(n, l, xi, gamma_g)
            phi2 = phi_fun(n, l+1, xi, gamma_g) if l+1 <= n else 0.0
            d[idx][l-1] = phi1 - phi2
    return b, d

b_pi, d_pi = compute_pi_coefficients(nsteps, r, lambda_r)

# **** Main loop over spatial refinements ****
nkmax = 5
hh = []; nn = []; eu = []; ru = []
eIu = []; rIu = []
ru.append(0.0); rIu.append(0.0)

for nk in range(nkmax):
    print("....... Refinement level : nk = ", nk)
    
    nps = pow(2, nk + 1)
    mesh = UnitSquareMesh(nps, nps)
    mesh_coarse = UnitSquareMesh(pow(2, nk), pow(2, nk))
    hh.append(mesh.hmax())
    
    Vh = FunctionSpace(mesh, 'CG', 1)
    W2h = FunctionSpace(mesh_coarse, 'CG', 2)
    nn.append(Vh.dim())
    
    v = TestFunction(Vh)
    u = TrialFunction(Vh)
    u_h = Function(Vh)
    
    # Exact solutions as Expressions
    u_ex = Expression(str2exp(u_str), t=0, alpha=alpha, degree=6, domain=mesh)
    Dt_alpha_u_ex = Expression(str2exp(Dt_alpha_u_str), t=0, Gamma_ratio=Gamma_ratio,
                               alpha=alpha, degree=6, domain=mesh)
    I_lambda_u_ex = Expression(str2exp(I_lambda_u_str), t=0, integral_factor=integral_factor,
                               alpha=alpha, lambda_r=lambda_r, degree=6, domain=mesh)
    u_ex_coarse = Expression(str2exp(u_str), t=0, alpha=alpha, degree=6, domain=mesh_coarse)
    
    # Initial condition at t=0
    u_old = interpolate(u_ex, Vh)
    u_history = [Function(Vh) for _ in range(nsteps + 1)]
    u_history[0].assign(u_old)
    
    # Time loop over graded steps
    E_u = 0.0
    E_Iu = 0.0
    
    for n in range(1, nsteps + 1):
        t = time_vec[n]
        dt = dt_vec[n - 1]
        print(f"    Time: t = {t:.4f}, dt = {dt:.4e}")
        
        C_alpha = dt ** (-alpha) / math.gamma(2 - alpha)
        
        # Update exact expressions
        u_ex.t = t
        Dt_alpha_u_ex.t = t
        I_lambda_u_ex.t = t
        u_ex_coarse.t = t
        
        # Exact forcing including the integral term
        f_ex = Dt_alpha_u_ex - div(K * grad(u_ex)) - gamma_coeff * I_lambda_u_ex
        
        # ---- L1 history sum for Caputo derivative ----
        history_sum = Function(Vh)
        history_sum.assign(Constant(0.0))
        for j in range(1, n):
            t_jm1 = time_vec[j - 1]
            t_j   = time_vec[j]
            dt_jm1 = dt_vec[j - 1]
            weight = ((t - t_jm1) ** (1 - alpha) - (t - t_j) ** (1 - alpha)) / (dt_jm1 * math.gamma(2 - alpha))
            diff = Function(Vh)
            diff.assign(u_history[j])
            diff.vector().axpy(-1.0, u_history[j - 1].vector())
            history_sum.vector().axpy(weight, diff.vector())
        
        # ---- PI history for the weakly singular integral term ----
        integral_rhs = Function(Vh)
        integral_rhs.assign(Constant(0.0))
        coeff_b = b_pi[n-1]
        integral_rhs.vector().axpy(coeff_b, u_history[0].vector())
        for l in range(1, n):
            coeff_d = d_pi[n-1][l-1]
            integral_rhs.vector().axpy(coeff_d, u_history[l].vector())
        
        # Coefficient for the current unknown
        coeff_curr = gamma_coeff * d_pi[n-1][n-1]   # appears with minus sign in LHS
        
        # Weak forms
        auv = (C_alpha - coeff_curr) * u * v * dx + dot(K * grad(u), grad(v)) * dx
        Fv = (f_ex + C_alpha * u_history[n - 1] - history_sum - gamma_coeff * integral_rhs) * v * dx
        
        bcU = DirichletBC(Vh, u_ex, 'on_boundary')
        u_curr = Function(Vh)
        solve(auv == Fv, u_curr, bcU)
        
        # Store solution
        u_history[n].assign(u_curr)
        u_h.assign(u_curr)
        
        # Quasi‑interpolant (simple interpolation on coarse mesh)
        Iu = interpolate(u_h, W2h)
        
        # Maximum‑in‑time H1 semi‑norm error
        err_u = assemble(dot(grad(u_ex) - grad(u_h), grad(u_ex) - grad(u_h)) * dx)
        err_Iu = assemble(dot(grad(u_ex_coarse) - grad(Iu), grad(u_ex_coarse) - grad(Iu)) * dx(domain=mesh_coarse))
        E_u = max(E_u, err_u)
        E_Iu = max(E_Iu, err_Iu)
    
    eu.append(np.sqrt(E_u))
    eIu.append(np.sqrt(E_Iu))
    
    if nk > 0:
        ru.append(np.log(eu[nk] / eu[nk - 1]) / np.log(hh[nk] / hh[nk - 1]))
        rIu.append(np.log(eIu[nk] / eIu[nk - 1]) / np.log(hh[nk] / hh[nk - 1]))

# Print error table
print('\n' + '='*50)
print('  DoF      h    e_1(u)   r_1(u)   e_1(Iu)  r_1(Iu)   ')
print('='*50)
for nk in range(nkmax):
    print('{:6d}  {:.4f} {:6.2e}  {:.3f}  {:6.2e}  {:.3f} '.format(
          nn[nk], hh[nk], eu[nk], ru[nk], eIu[nk], rIu[nk]))
print('='*50)