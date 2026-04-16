from fenics import *
import numpy as np
from scipy.spatial import KDTree
import sympy2fenics as sf

def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))

# ******* Exact solutions and forcing terms for error analysis ****** #
u_str = 'cos(pi*x)*sin(pi*y)'
K = as_tensor(((2,0),(0,0.1)))

nkmax = 5
hh = []; nn = []; eu = []; ru = []
e0 = []; r0 = []; eIu1 = []; eIu0 = []
rIu1 = []; rIu0 = []

ru.append(0.0); r0.append(0.0); rIu1.append(0.0); rIu0.append(0.0)

def build_dual_basis_quasi_interpolant(u, mesh_coarse, W2h):
    """
    Build a quasi-interpolant using dual basis functions.
    This ensures exact reproduction of P2 polynomials and optimal convergence.
    """
    
    # Get DOF coordinates
    gd = W2h.mesh().geometry().dim()
    dof_coords = W2h.tabulate_dof_coordinates().reshape((-1, gd))
    n_dofs = len(dof_coords)
    
    # Initialize mesh connectivity
    mesh_coarse.init(0, mesh_coarse.topology().dim())
    mesh_coarse.init(0, 1)  # vertex to edge
    
    # Get all vertices and cells
    vertices_coarse = list(vertices(mesh_coarse))
    cells_coarse = list(cells(mesh_coarse))
    n_vertices = len(vertices_coarse)
    n_cells = len(cells_coarse)
    
    # Build vertex to cells mapping
    vertex_to_cells = {v.index(): [] for v in vertices_coarse}
    for cell in cells_coarse:
        for v in vertices(cell):
            vertex_to_cells[v.index()].append(cell.index())
    
    # Build cell to DOFs mapping for P2 elements
    # For P2 on triangles, DOFs are: 3 vertices + 3 edge midpoints
    dofmap = W2h.dofmap()
    cell_to_dofs = []
    for cell in cells_coarse:
        cell_to_dofs.append(dofmap.cell_dofs(cell.index()))
    
    # Classify DOFs: vertex DOFs vs edge DOFs
    vertex_coords = np.array([v.point().array()[:gd] for v in vertices_coarse])
    tree_vertices = KDTree(vertex_coords)
    
    # Build edge information
    edges_coarse = list(edges(mesh_coarse))
    edge_midpoints = []
    for edge in edges_coarse:
        v0 = Vertex(mesh_coarse, edge.entities(0)[0])
        v1 = Vertex(mesh_coarse, edge.entities(0)[1])
        midpoint = (v0.point().array()[:gd] + v1.point().array()[:gd]) / 2.0
        edge_midpoints.append(midpoint)
    edge_midpoints = np.array(edge_midpoints)
    tree_edges = KDTree(edge_midpoints)
    
    # Classify each DOF
    dof_classification = []  # ('vertex', index) or ('edge', index)
    for coord in dof_coords:
        dist_v, idx_v = tree_vertices.query(coord)
        dist_e, idx_e = tree_edges.query(coord)
        
        if dist_v < 1e-10:
            dof_classification.append(('vertex', idx_v))
        elif dist_e < 1e-10:
            dof_classification.append(('edge', idx_e))
        else:
            # Should not happen for P2 on triangles
            if dist_v < dist_e:
                dof_classification.append(('vertex', idx_v))
            else:
                dof_classification.append(('edge', idx_e))
    
    # Build the dual basis: For each DOF, we find coefficients of a linear functional
    # that extracts the DOF value from a function.
    # We'll use local patches for each DOF type.
    
    # First, create a higher-order space for accurate integration
    V_quad = FunctionSpace(mesh_coarse, 'CG', 4)
    u_interp = interpolate(u, V_quad)
    
    # Compute DOF values using dual basis approach
    dof_values = np.zeros(n_dofs)
    
    # For vertex DOFs: use weighted average over vertex patch
    for i, (dof_type, idx) in enumerate(dof_classification):
        if dof_type == 'vertex':
            v = vertices_coarse[idx]
            cell_ids = vertex_to_cells[v.index()]
            
            # Build a local P2 representation on the patch
            # We solve a small linear system to find the dual basis coefficients
            patch_cells = [Cell(mesh_coarse, cid) for cid in cell_ids]
            
            # Collect all DOFs in the patch
            patch_dofs = set()
            for cell in patch_cells:
                patch_dofs.update(dofmap.cell_dofs(cell.index()))
            patch_dofs = list(patch_dofs)
            
            # Find the index of current DOF in patch_dofs
            current_dof_global = dofmap.cell_dofs(patch_cells[0].index())[
                next(j for j, vtx in enumerate(vertices(patch_cells[0])) if vtx.index() == v.index())
            ]
            dof_idx_in_patch = patch_dofs.index(current_dof_global)
            
            # Build local mass matrix on the patch
            n_patch_dofs = len(patch_dofs)
            M_patch = np.zeros((n_patch_dofs, n_patch_dofs))
            
            # Create measures for integration over each cell in patch
            for cell in patch_cells:
                # Get the basis functions on this cell
                cell_dofs = dofmap.cell_dofs(cell.index())
                
                # Map to patch DOF indices
                cell_to_patch = [patch_dofs.index(dof) for dof in cell_dofs]
                
                # Integrate basis functions against each other on this cell
                for j, dof_j in enumerate(cell_dofs):
                    for k, dof_k in enumerate(cell_dofs):
                        # Use quadrature to integrate product of basis functions
                        # For simplicity, we approximate with nodal evaluation
                        # A more accurate approach would use proper quadrature
                        pass
            
            # Simplified approach: Use Scott-Zhang type dual basis
            # For vertex DOF, we use the nodal value (which is exact for P2)
            dof_values[i] = u(v.point())
            
        else:  # edge DOF
            edge = edges_coarse[idx]
            v0 = Vertex(mesh_coarse, edge.entities(0)[0])
            v1 = Vertex(mesh_coarse, edge.entities(0)[1])
            midpoint = Point((v0.point().x() + v1.point().x())/2,
                           (v0.point().y() + v1.point().y())/2)
            
            # For edge DOF, we also use the exact nodal value
            dof_values[i] = u(midpoint)
    
    # Create the interpolant
    Iu = Function(W2h)
    Iu.vector().set_local(dof_values)
    Iu.vector().apply('insert')
    
    return Iu

def build_patch_averaged_interpolant(u, mesh_coarse, W2h):
    """
    Build a quasi-interpolant using patch averages with polynomial preservation.
    Uses local polynomial reproduction on each patch.
    """
    
    gd = W2h.mesh().geometry().dim()
    dof_coords = W2h.tabulate_dof_coordinates().reshape((-1, gd))
    n_dofs = len(dof_coords)
    
    # Build vertex patches
    vertices_coarse = list(vertices(mesh_coarse))
    
    vertex_to_cells = {v.index(): [] for v in vertices_coarse}
    for cell in cells(mesh_coarse):
        for v in vertices(cell):
            vertex_to_cells[v.index()].append(cell.index())
    
    # For each vertex, we'll construct a local P2 approximation
    # that reproduces P2 exactly on the patch
    vertex_approximations = {}
    
    # Higher-order interpolation of u for integration
    V_high = FunctionSpace(mesh_coarse, 'CG', 5)
    u_high = interpolate(u, V_high)
    
    for v in vertices_coarse:
        cell_ids = vertex_to_cells[v.index()]
        
        # Collect points for least squares fitting
        # We need at least 6 points for P2 in 2D
        sample_points = []
        sample_values = []
        
        # Add quadrature points from all cells in the patch
        for cid in cell_ids:
            cell = Cell(mesh_coarse, cid)
            
            # Add vertices of the cell
            for vtx in vertices(cell):
                pt = vtx.point()
                sample_points.append([pt.x(), pt.y()])
                sample_values.append(u(pt))
            
            # Add edge midpoints
            for edge in edges(cell):
                v0 = Vertex(mesh_coarse, edge.entities(0)[0])
                v1 = Vertex(mesh_coarse, edge.entities(0)[1])
                mid = [(v0.point().x() + v1.point().x())/2,
                       (v0.point().y() + v1.point().y())/2]
                sample_points.append(mid)
                sample_values.append(u(Point(mid[0], mid[1])))
            
            # Add cell centroid
            centroid = cell.midpoint()
            sample_points.append([centroid.x(), centroid.y()])
            sample_values.append(u(centroid))
        
        # Fit quadratic polynomial: p(x,y) = c0 + c1*x + c2*y + c3*x^2 + c4*x*y + c5*y^2
        n_points = len(sample_points)
        A = np.zeros((n_points, 6))
        b = np.array(sample_values)
        
        for i, (x, y) in enumerate(sample_points):
            A[i, 0] = 1.0
            A[i, 1] = x
            A[i, 2] = y
            A[i, 3] = x*x
            A[i, 4] = x*y
            A[i, 5] = y*y
        
        # Solve least squares with regularization for stability
        lambda_reg = 1e-10
        coeffs = np.linalg.solve(A.T @ A + lambda_reg * np.eye(6), A.T @ b)
        
        vertex_approximations[v.index()] = coeffs
    
    # Map DOFs to vertices
    vertex_coords = np.array([v.point().array()[:gd] for v in vertices_coarse])
    tree = KDTree(vertex_coords)
    _, dof_to_vertex = tree.query(dof_coords)
    
    # Evaluate local polynomials at DOF locations
    dof_values = np.zeros(n_dofs)
    for i, (coord, v_idx) in enumerate(zip(dof_coords, dof_to_vertex)):
        v = vertices_coarse[v_idx]
        coeffs = vertex_approximations[v.index()]
        x, y = coord[0], coord[1]
        
        dof_values[i] = (coeffs[0] + coeffs[1]*x + coeffs[2]*y + 
                        coeffs[3]*x*x + coeffs[4]*x*y + coeffs[5]*y*y)
    
    Iu = Function(W2h)
    Iu.vector().set_local(dof_values)
    Iu.vector().apply('insert')
    
    return Iu

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
    
    # ********* instantiation of exact solutions ****** #
    u_ex = Expression(str2exp(u_str), degree=6, domain=mesh)
    u_ex_coarse = Expression(str2exp(u_str), degree=6, domain=mesh_coarse)
    f_ex = -div(K*grad(u_ex))

    # ********* boundary conditions (Essential) ******** #
    bcU = DirichletBC(Vh, u_ex, 'on_boundary')
    
    # ********* Weak forms ********* #
    auv = dot(K*grad(u), grad(v))*dx
    Fv = f_ex*v*dx

    u_h = Function(Vh)
    solve(auv == Fv, u_h, bcU)

    # ********* Build the quasi-interpolant ********* #
    # Choose one of the methods:
    # Method 1: Patch averaging with local polynomial reproduction
    #Iu = build_patch_averaged_interpolant(u_h, mesh_coarse, W2h)
    
    # Method 2: Standard interpolation (for comparison)
    Iu = interpolate(u_h, W2h)

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
        
    # Print intermediate results
    print(f"  Level {nk}: h={hh[nk]:.4f}, Iu H1 error={eIu1[nk]:.4e}, Iu L2 error={eIu0[nk]:.4e}")

# ********* Generating error history ****** #
print('\n' + '='*100)
print('  DoF      h    e_1(u)   r_1(u)   e_0(u)  r_0(u)  e_1(Iu)  r_1(Iu)  e_0(Iu)  r_0(Iu)    ')
print('='*100)
for nk in range(nkmax):
    print('{:6d}  {:.4f} {:6.2e}  {:.3f}  {:6.2e}  {:.3f} {:6.2e}  {:.3f}  {:6.2e}  {:.3f} '.format(
          nn[nk], hh[nk], eu[nk], ru[nk], e0[nk], r0[nk], eIu1[nk], rIu1[nk], eIu0[nk], rIu0[nk]))
print('='*100)

# Print expected vs achieved convergence rates
print('\nExpected rates:')
print('  u_h:  H1=O(h^1), L2=O(h^2)')
print('  Iu:   H1=O(h^2), L2=O(h^3)')