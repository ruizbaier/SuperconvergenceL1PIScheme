from fenics import *
import numpy as np
from scipy.spatial import KDTree
import sympy2fenics as sf

def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))

# ******* Exact solutions and forcing terms for error analysis ****** #
u_str = 'cos(pi*x)*sin(pi*y)*cos(pi*z)'  # 3D test function
K = as_tensor(((2,0,0),(0,1,0),(0,0,0.5)))

nkmax = 4  # Reduced for 3D due to computational cost
hh = []; nn = []; eu = []; ru = []
e0 = []; r0 = []; eIu1 = []; eIu0 = []
rIu1 = []; rIu0 = []

ru.append(0.0); r0.append(0.0); rIu1.append(0.0); rIu0.append(0.0)

def build_patch_averaged_interpolant_3D(u, mesh_coarse, W2h):
    """
    Build a quasi-interpolant using patch averages with polynomial preservation in 3D.
    Uses local polynomial reproduction on each vertex patch.
    For P2 in 3D, we have 10 DOFs per tetrahedron:
    - 4 vertices
    - 6 edge midpoints
    """
    
    gd = W2h.mesh().geometry().dim()  # Should be 3
    dof_coords = W2h.tabulate_dof_coordinates().reshape((-1, gd))
    n_dofs = len(dof_coords)
    
    # Build vertex patches
    vertices_coarse = list(vertices(mesh_coarse))
    
    vertex_to_cells = {v.index(): [] for v in vertices_coarse}
    for cell in cells(mesh_coarse):
        for v in vertices(cell):
            vertex_to_cells[v.index()].append(cell.index())
    
    # For each vertex, construct a local P2 approximation
    vertex_approximations = {}
    
    for v_idx, v in enumerate(vertices_coarse):
        if v_idx % 100 == 0:
            print(f"    Processing vertex {v_idx}/{len(vertices_coarse)}")
            
        cell_ids = vertex_to_cells[v.index()]
        
        # Collect points for least squares fitting
        # For P2 in 3D, we have 10 basis functions: 
        # 1, x, y, z, x^2, y^2, z^2, xy, xz, yz
        sample_points = []
        sample_values = []
        
        for cid in cell_ids:
            cell = Cell(mesh_coarse, cid)
            
            # Add vertices of the tetrahedron (4 points)
            for vtx in vertices(cell):
                pt = vtx.point()
                sample_points.append([pt.x(), pt.y(), pt.z()])
                sample_values.append(u(pt))
            
            # Add edge midpoints (6 points)
            edges_cell = edges(cell)
            for edge in edges_cell:
                v0 = Vertex(mesh_coarse, edge.entities(0)[0])
                v1 = Vertex(mesh_coarse, edge.entities(0)[1])
                mid = [(v0.point().x() + v1.point().x())/2,
                       (v0.point().y() + v1.point().y())/2,
                       (v0.point().z() + v1.point().z())/2]
                sample_points.append(mid)
                sample_values.append(u(Point(mid[0], mid[1], mid[2])))
            
            # Add face centroids (4 points) - optional for better conditioning
            for face in faces(cell):
                vertices_face = face.entities(0)
                if len(vertices_face) == 3:  # Triangle face
                    v0 = Vertex(mesh_coarse, vertices_face[0])
                    v1 = Vertex(mesh_coarse, vertices_face[1])
                    v2 = Vertex(mesh_coarse, vertices_face[2])
                    centroid = [(v0.point().x() + v1.point().x() + v2.point().x())/3,
                               (v0.point().y() + v1.point().y() + v2.point().y())/3,
                               (v0.point().z() + v1.point().z() + v2.point().z())/3]
                    sample_points.append(centroid)
                    sample_values.append(u(Point(centroid[0], centroid[1], centroid[2])))
            
            # Add cell centroid (1 point)
            centroid = cell.midpoint()
            sample_points.append([centroid.x(), centroid.y(), centroid.z()])
            sample_values.append(u(centroid))
        
        # Remove duplicate points (from shared edges/faces)
        unique_points = []
        unique_values = []
        tolerance = 1e-10
        
        for pt, val in zip(sample_points, sample_values):
            is_duplicate = False
            for existing_pt in unique_points:
                if np.linalg.norm(np.array(pt) - np.array(existing_pt)) < tolerance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_points.append(pt)
                unique_values.append(val)
        
        sample_points = unique_points
        sample_values = unique_values
        
        # Fit quadratic polynomial in 3D: 10 basis functions
        # p(x,y,z) = c0 + c1*x + c2*y + c3*z + c4*x^2 + c5*y^2 + c6*z^2 + 
        #            c7*x*y + c8*x*z + c9*y*z
        n_points = len(sample_points)
        n_basis = 10
        
        A = np.zeros((n_points, n_basis))
        b = np.array(sample_values)
        
        for i, (x, y, z) in enumerate(sample_points):
            A[i, 0] = 1.0
            A[i, 1] = x
            A[i, 2] = y
            A[i, 3] = z
            A[i, 4] = x*x
            A[i, 5] = y*y
            A[i, 6] = z*z
            A[i, 7] = x*y
            A[i, 8] = x*z
            A[i, 9] = y*z
        
        # Solve least squares with regularization for stability
        lambda_reg = 1e-10
        try:
            coeffs = np.linalg.solve(A.T @ A + lambda_reg * np.eye(n_basis), A.T @ b)
        except:
            # Fallback to pseudo-inverse if solve fails
            coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
        
        vertex_approximations[v.index()] = coeffs
    
    # Classify DOFs in 3D
    # Build KD-trees for different DOF types
    vertex_coords = np.array([v.point().array()[:gd] for v in vertices_coarse])
    tree_vertices = KDTree(vertex_coords)
    
    # Build edge information
    mesh_coarse.init(1)  # Initialize edges
    edges_coarse = list(edges(mesh_coarse))
    edge_midpoints = []
    for edge in edges_coarse:
        v0 = Vertex(mesh_coarse, edge.entities(0)[0])
        v1 = Vertex(mesh_coarse, edge.entities(0)[1])
        midpoint = [(v0.point().x() + v1.point().x())/2,
                   (v0.point().y() + v1.point().y())/2,
                   (v0.point().z() + v1.point().z())/2]
        edge_midpoints.append(midpoint)
    edge_midpoints = np.array(edge_midpoints)
    tree_edges = KDTree(edge_midpoints) if len(edge_midpoints) > 0 else None
    
    # Build face information (for P2 on tetrahedra, no face DOFs, but included for completeness)
    mesh_coarse.init(2)  # Initialize faces
    faces_coarse = list(faces(mesh_coarse))
    face_centroids = []
    for face in faces_coarse:
        vertices_face = face.entities(0)
        if len(vertices_face) == 3:  # Triangle face
            v0 = Vertex(mesh_coarse, vertices_face[0])
            v1 = Vertex(mesh_coarse, vertices_face[1])
            v2 = Vertex(mesh_coarse, vertices_face[2])
            centroid = [(v0.point().x() + v1.point().x() + v2.point().x())/3,
                       (v0.point().y() + v1.point().y() + v2.point().y())/3,
                       (v0.point().z() + v1.point().z() + v2.point().z())/3]
            face_centroids.append(centroid)
    face_centroids = np.array(face_centroids)
    tree_faces = KDTree(face_centroids) if len(face_centroids) > 0 else None
    
    # Map each DOF to the nearest vertex
    # For P2 on tetrahedra, all DOFs are on vertices or edges
    dof_to_vertex = []
    for coord in dof_coords:
        dist_v, idx_v = tree_vertices.query(coord)
        
        # Check if it's an edge midpoint
        if tree_edges is not None:
            dist_e, idx_e = tree_edges.query(coord)
            if dist_e < 1e-8:
                # It's an edge midpoint - still map to nearest vertex
                dof_to_vertex.append(idx_v)
            else:
                dof_to_vertex.append(idx_v)
        else:
            dof_to_vertex.append(idx_v)
    
    # Evaluate local polynomials at DOF locations
    dof_values = np.zeros(n_dofs)
    for i, (coord, v_idx) in enumerate(zip(dof_coords, dof_to_vertex)):
        v = vertices_coarse[v_idx]
        coeffs = vertex_approximations[v.index()]
        x, y, z = coord[0], coord[1], coord[2]
        
        dof_values[i] = (coeffs[0] + coeffs[1]*x + coeffs[2]*y + coeffs[3]*z + 
                        coeffs[4]*x*x + coeffs[5]*y*y + coeffs[6]*z*z +
                        coeffs[7]*x*y + coeffs[8]*x*z + coeffs[9]*y*z)
    
    Iu = Function(W2h)
    Iu.vector().set_local(dof_values)
    Iu.vector().apply('insert')
    
    return Iu

# Main loop
for nk in range(nkmax):
    print(f"\n....... Refinement level : nk = {nk} .......")
    
    nps = pow(2, nk+1)
    
    # Create 3D meshes
    mesh = UnitCubeMesh(nps, nps, nps)
    mesh_coarse = UnitCubeMesh(pow(2, nk), pow(2, nk), pow(2, nk))
    hh.append(mesh.hmax())

    # ********* Finite dimensional spaces ********* #
    Vh = FunctionSpace(mesh, 'CG', 1)
    W2h = FunctionSpace(mesh_coarse, 'CG', 2)
    nn.append(Vh.dim())
    
    print(f"  Fine mesh DOFs: {Vh.dim()}, Coarse mesh DOFs: {W2h.dim()}")
    
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
    
    print("  Solving fine mesh problem...")
    solve(auv == Fv, u_h, bcU)

    # ********* Build the quasi-interpolant ********* #
    print("  Building quasi-interpolant...")
    Iu = build_patch_averaged_interpolant_3D(u_h, mesh_coarse, W2h)

    #Iu = interpolate(u_h,W2h) #build_patch_averaged_interpolant_3D(u_h, mesh_coarse, W2h)
    

    # ********* Computing errors ****** #
    print("  Computing errors...")
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
    
    print(f"  Results: u_h H1={eu[nk]:.4e}, L2={e0[nk]:.4e}")
    print(f"           Iu H1={eIu1[nk]:.4e}, L2={eIu0[nk]:.4e}")

# ********* Generating error history ****** #
print('\n' + '='*120)
print('  DoF      h         e_1(u)    r_1(u)   e_0(u)    r_0(u)   e_1(Iu)   r_1(Iu)  e_0(Iu)   r_0(Iu)')
print('='*120)
for nk in range(nkmax):
    print('{:8d}  {:.4f}  {:6.2e}  {:.3f}  {:6.2e}  {:.3f}  {:6.2e}  {:.3f}  {:6.2e}  {:.3f}'.format(
          nn[nk], hh[nk], eu[nk], ru[nk], e0[nk], r0[nk], eIu1[nk], rIu1[nk], eIu0[nk], rIu0[nk]))
print('='*120)

# Print expected vs achieved convergence rates
print('\nTheoretical convergence rates:')
print('  u_h:  H1=O(h^1), L2=O(h^2)')
print('  Iu:   H1=O(h^2), L2=O(h^3)')
print('\nAchieved rates for Iu (last refinement):')
if nkmax > 1:
    print(f'  H1: {rIu1[-1]:.2f}, L2: {rIu0[-1]:.2f}')