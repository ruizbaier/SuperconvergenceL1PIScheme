from fenics import *
import sympy2fenics as sf
import math
import numpy as np


fileU = XDMFFile("outputs/fullProblem-noExact-2D.xdmf")
fileU.parameters["functions_share_mesh"] = True
fileU.parameters["flush_output"] = True

fileIU = XDMFFile("outputs/fullProblem-noExact-2D-iu.xdmf")
fileIU.parameters["functions_share_mesh"] = True
fileIU.parameters["flush_output"] = True

# ******* Exact solutions and forcing terms for error analysis ****** #
Tfinal = 1.0
nsteps = 100          # total number of intervals
alpha  = 0.75         # fractional order
lambda_r = 0.5        # order of weakly singular integral (0 < lambda < 1)
gamma_coeff = 0.5    # coefficient in front of the integral term 


K = as_tensor(((1, 0.1), (0.2, 0.05)))

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

# mesh  ****
mesh = UnitSquareMesh(80, 80)
mesh_coarse = UnitSquareMesh(40, 40)

# right-hand side function 
f_ex = Expression('100*sin(pi*x[0]*cos(t))*sin(pi*x[1]*exp(t))*cos(pi*(x[0]+x[1])*exp(-t))', t=0.0, degree=6, domain=mesh)

Vh = FunctionSpace(mesh, 'CG', 1)
W2h = FunctionSpace(mesh_coarse, 'CG', 2)
    
v = TestFunction(Vh)
u = TrialFunction(Vh)
u_h = Function(Vh)
     
# Initial condition at t=0
u_old = interpolate(Constant(0.0), Vh)
u_history = [Function(Vh) for _ in range(nsteps + 1)]
u_history[0].assign(u_old)
    
# Time loop over graded steps

for n in range(1, nsteps + 1):
    t = time_vec[n]
    dt = dt_vec[n - 1]
    print(f"    Time: t = {t:.4f}, dt = {dt:.4e}")
        
    C_alpha = dt ** (-alpha) / math.gamma(2 - alpha)
        
    # Update right-hand side  
    f_ex.t = t
    
         
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
        
    bcU = DirichletBC(Vh, Constant(0.0), 'on_boundary')
    u_curr = Function(Vh)
    solve(auv == Fv, u_curr, bcU)
        
    # Store solution
    u_history[n].assign(u_curr)
    u_h.assign(u_curr)
        
    # Quasi‑interpolant (simple interpolation on coarse mesh)
    
    Iu = interpolate(u_h, W2h)
    
    u_h.rename("u", "u")
    fileU.write(u_h, t)
    Iu.rename("Iu", "Iu")
    fileIU.write(Iu, t)
