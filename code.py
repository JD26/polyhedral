import numpy as np

import gmsh
# gmsh.info/

import ufl

from dolfinx.io import gmshio, XDMFFile
from dolfinx.plot import vtk_mesh
from dolfinx.fem.petsc import LinearProblem
from dolfinx import fem, mesh, default_scalar_type, plot

from mpi4py import MPI

import pyvista

from scipy.spatial import ConvexHull
from pypoman import compute_polytope_vertices
# github.com/stephane-caron/pypoman
# or.stackexchange.com/questions/4540/how-to-find-all-vertices-of-a-polyhedron

import matplotlib.pyplot as plt

# Polyhedral examples

pentagonal_prism = {
	"K": 7,
	"A": [[-1., 0.5, 0.], \
		  [-1., -0.5, 0.], \
		  [0., -1., 0.], \
		  [0., 0., -1.], \
		  [0., 0., 1.], \
		  [0., 1., 0.], \
		  [1., 0, 0.]],
	"b": [1., 1. , 1. , 1., 1., 1., 1.],
	"nv": 10,
	"vertices": [[4, 6, 5], [4, 5, 0], [4, 0, 1], [4, 1, 2], [4, 2, 6], \
				 [3, 6, 5], [3, 5, 0], [3, 0, 1], [3, 1, 2], [3, 2, 6]]
}

hexagonal_prism = {
	"K": 8,
	"A": [[-1., 0.5, 0.], \
		  [-1., -0.5, 0.], \
		  [0, -1., 0.], \
		  [0, 0., -1.], \
		  [0., 0., 1.], \
		  [0., 1., 0.], \
		  [1., 0.5, 0.], \
		  [1., -0.5, 0.]],
	"b": [1., 1., 1. , 1. , 1., 1., 1., 1.],
	"nv": 12,
	"vertices": [[3, 7, 6], [3, 6, 5], [3, 5, 0], \
				 [3, 0, 1], [3, 1, 2], [3, 2, 7], \
				 [4, 7, 6], [4, 6, 5], [4, 5, 0], \
				 [4, 0, 1], [4, 1, 2], [4, 2, 7]] 	
}

# minimum variance : iter = 10
tetrahedron = {
	"K": 4,
	"A": [[0., 0., -1.], \
		  [1., 0., 1.], \
		  [0., 1., 1.], \
		  [-1., -1., 1.]],
	"b": [1., 1., 1., 1.],
	"nv": 4,
	"vertices": [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 2, 3]]
}

# minimum variance : iter = 77
cube = {
	"K": 6,
	"A": [[-1., -0.2, 0.], \
		  [0., -1., 0.], \
		  [0., 0., -1.], \
		  [0., 0., 1.], \
		  [0., 1., 0.], \
		  [1., 0.5, 0.]],
	"b": [1. , 1. , 1., 1., 1., 1.],
	"nv": 8, 
	"vertices": [[3, 1, 5], [3, 5, 4], [3, 4, 0], [3, 0, 1], \
				 [2, 1, 5], [2, 5, 4], [2, 4, 0], [2, 0, 1]]
}

dodecahedron = {
	"K": 12,
	"A": [[1., 0.5, 0.], \
		  [-1., -0.5, 0.], \
		  [1., -0.5, 0.], \
		  [-1., 0.5, 0.], \
		  [0., 1., 0.5], \
		  [0., -1., -0.5], \
		  [0., 1., -0.6], \
		  [0., -1., 0.6], \
		  [0.5, 0., 1.], \
		  [-0.6, 0., -1.], \
		  [-0.6, 0., 1.], \
		  [0.5, 0., -1.]],
	"b": [1.2, 1., 1., 1., 1.2, 1., 1., 1., 1.2, 1., 1., 1.],
	"nv": 20,
	"vertices": [[0, 2, 11], [0, 2, 8], [1, 3, 9], [1, 3, 10], \
				 [4, 6, 0], [4, 6, 3], [5, 7, 2], [5, 7, 1], \
				 [8, 10, 4], [8, 10, 7], [9, 11, 5], [9, 11, 6], \
				 [0, 8, 4], [2, 8, 7], [1, 10, 7], [3, 10, 4], \
				 [0, 11, 6], [2, 11, 5], [3, 9, 6], [1, 9, 5]] 	
}

chamfered_tetrahedron = {
	"K": 10,
	"A": [[1., 0., 0.], \
		  [0., 1., 0.], \
		  [0., 0., 1.], \
		  [-0.5, 0.5, 0.5], \
		  [0.5, -0.5, 0.5], \
		  [0.5, 0.5, -0.5], \
		  [-1., 0., 0.], \
		  [0., -1., 0.], \
		  [0., 0., -1.], \
		  [-0.5, -0.5, -0.5]],
	"b": [1. , 1. , 1., 1., 1., 1., 1., 1., 1., 1.],
	"nv": 16,
	"vertices": [[0, 1, 2], [1, 2, 3], [2, 0, 4], [0, 1, 5], \
				 [1, 3, 6], [2, 3, 6], \
				 [2, 4, 7], [0, 4, 7], \
				 [0, 5, 8], [1, 5, 8], \
				 [0, 7, 8], [1, 6, 8], [2, 6, 7], \
				 [6, 7, 9], [6, 8, 9], [7, 8, 9]]
}

chamfered_cube = {
	"K": 18,
	"A": [[1., 0., 0.], \
		  [-1., 0., 0.], \
		  [0., 1., 0.], \
		  [0., -1., 0.], \
		  [0., 0., 1.], \
		  [0., 0., -1.], \
		  [1., 1., 0.], \
		  [1., -1., 0.], \
		  [-1., 1., 0.], \
		  [-1., -1., 0.], \
		  [1., 0., 1.], \
		  [1., 0., -1.], \
		  [-1., 0., 1.], \
		  [-1., 0., -1.], \
		  [0., 1., 1.], \
		  [0., 1., -1.], \
		  [0., -1., 1.], \
		  [0., -1., -1.]],
	"b": [0.8, 0.9, 0.8, 0.9, 0.6, 0.6, 1.1, 1., 1., 1.4, 1., 1., 1., 1., 1., 1., 1., 1.],
	"vertices": [] # vertices need to be written here...
}

octagonal_prism = {
	"K": 10,
	"A": [[-1., 0.5, 0.], \
		  [-1., -0.5, 0.], \
		  [-0.5, -1., 0.], \
		  [0.5, -1., 0.], \
		  [1., -0.5, 0.], \
		  [1., 0.5, 0.], \
		  [0.5, 1., 0.], \
		  [-0.5, 1., 0.], \
		  [0, 0., -1.], \
		  [0., 0., 1.]],
	"b": [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
	"vertices": [[8, 0, 1], [8, 1, 2], [8, 2, 3], [8, 3, 4], \
				 [8, 4, 5], [8, 5, 6], [8, 6, 7], [8, 7, 0], \
				 [9, 0, 1], [9, 1, 2], [9, 2, 3], [9, 3, 4], \
				 [9, 4, 5], [9, 5, 6], [9, 6, 7], [9, 7, 0]] 	
}

def Transform(vertices, n, a):
	# Example :
	# Transform(self.vertices, np.array([1., 1., 1.])/np.sqrt(3), 1.)
	Y = []
	vt = np.array([0., 0., -a/n[2]])
	for x in vertices:
		Y.append(x + (a - np.inner(x, n))*n + vt) 
	Y = np.array(Y)
	r1 = np.sqrt(n[0]**2 + n[1]**2)
	r2 = np.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
	cost = n[2]/r2
	sint = r1/r2
	u1 = n[1]/r1
	u2 = -n[0]/r1
	u12 = u1*u2 
	R1 = [cost + (1.-cost)*u1**2, u12*(1.-cost), u2*sint]
	R2 = [u12*(1.-cost), cost + (1.-cost)*u2**2, -u1*sint]
	R3 = [-u2*sint, u1*sint, cost]
	R = np.array([R1, R2, R3])
	vxy = np.matmul(R,Y.transpose()).transpose()
	return vxy[:, [0, 1]]

def CreateClosedHalfPlane(gmsh, n, a, k, c = 50):
	# create a closed half-plane H = {x | <n,x> <= a}
	# 'n' is a u unit vector
	# 'a' must be positive ==> 0 in H
	# tag k must be greater than or equal to 1
	p, q = a*n, -k*c*n
	gmsh.model.occ.addCylinder(p[0], p[1], p[2], q[0], q[1], q[2],  k*c, tag = k)
	
def IntersectionOf(gmsh, K):
	# intersect the closed half-planes
	# K is the number of closed half-spaces
	# the tags are K+1, ..., 2K-1 
	gmsh.model.occ.intersect([(3, 1)], [(3, 2)], K+1)
	for i in range(1, K-1):
		#print(K+i, 2+i, K+i+1)
		gmsh.model.occ.intersect([(3, K+i)], [(3, 2+i)], tag = (K+i+1))
	gmsh.model.occ.synchronize()

def AddNames(gmsh, K):
	# 1 volume = domain 
	gmsh.model.addPhysicalGroup(3, [2*K-1], 2*K)
	gmsh.model.setPhysicalName(3, 2*K, "Domain")
	# K surfaces = faces
	surfaces = gmsh.model.occ.getEntities(2)
	gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], 2*K+1)
	gmsh.model.setPhysicalName(2, 2*K+1, "Boundary")
	gmsh.model.occ.synchronize()
		
def SetSizeMesh(gmsh, size):
	gmsh.model.mesh.setSize(gmsh.model.getEntities(0), size)
	#gmsh.option.setNumber("Mesh.CharacteristicLengthMin", size)
	#gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size)
	gmsh.model.occ.synchronize()
	
def CreatePolyhedral(gmsh, A, b, K, size):
	#print('Matrix A :', A)
	#print('Vector b: ', b)
	for k in range(1, K+1):
		CreateClosedHalfPlane(gmsh, A[k-1], b[k-1], k)
	IntersectionOf(gmsh, K)
	AddNames(gmsh, K)
	SetSizeMesh(gmsh, size)
	gmsh.model.mesh.generate() #create the mesh	

def Normalize(A, b):
	# This function normalizes
	# A and b such that each
	# row of A has 2-norm
	# iquals to 1. 
	normA = np.linalg.norm(A, ord = 2, axis = 1)
	An = A/normA[:,None]
	bn = b/normA
	return An, bn

def PlotPolyhedral(pl, mesh_size):
	A = np.array(pl["A"])
	b = np.array(pl["b"])
	A, b = Normalize(A, b)
	gmsh.initialize()
	gmsh.option.setNumber("General.Terminal", 0) # Hide info from gmesh
	CreatePolyhedral(gmsh, A, b, pl["K"], mesh_size)
	gmsh.fltk.run()

def VolumeSc(A, b, nde = 14):
	vertices = compute_polytope_vertices(A, b)
	return np.round(ConvexHull(vertices).volume, decimals = nde)
	
def VolumeFx(domain):
	return fem.assemble_scalar(fem.form(Constant(domain, 1.) * ufl.dx))

def GetDomain(K, A, b, size_mesh, plot = False): 
	gmsh.initialize()
	gmsh.option.setNumber("General.Terminal", 0) # Hide info from gmesh
	CreatePolyhedral(gmsh, A, b, K, size_mesh)
	if plot : gmsh.fltk.run() # Plot the mesh
	# Convert a gmsh mesh to fenics domain
	domain, cell_tags, facet_tags = \
		gmshio.model_to_mesh(model = gmsh.model,
							comm = MPI.COMM_WORLD,
							rank = 0, # For parallel processes
							gdim = 3)	
	gmsh.clear()
	gmsh.finalize()
	return domain

def GetfValue(domain, value):
	return fem.Constant(domain, default_scalar_type(value))

def GetfFunction(domain):
	x = ufl.SpatialCoordinate(domain)
	Cte = fem.Constant(domain, default_scalar_type(0.5))
	f = -10.*ufl.exp(-((x[0]-Cte)**2 + x[1]**2 + x[2]**2)) 
	return f

def GetBoundaryFunc(domain, V):
	domain.topology.create_connectivity(d0 = 2, d1 = 3)
	boundary_facets = mesh.exterior_facet_indices(domain.topology)
	return fem.locate_dofs_topological(V, 2, boundary_facets)

def GetuDValue(domain, V, value):
	uD = default_scalar_type(value)
	boundary_dofs = GetBoundaryFunc(domain, V)
	return fem.dirichletbc(uD, boundary_dofs, V)

def GetuDFunction(domain, V):
	uD = fem.Function(V)
	uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2 + x[2]**2)
	boundary_dofs = GetBoundaryFunc(domain, V)
	return fem.dirichletbc(uD, boundary_dofs)

def SolvePDE01(domain):
	# Solve PDE

	V = fem.FunctionSpace(domain, ("Lagrange", 1))
	f = GetfValue(domain, -6)
	#f = GetfFunction(domain)
	
	bc = GetuDValue(domain, V, 0.)
	#bc = GetuDFunction(domain, V)

	u = ufl.TrialFunction(V)
	v = ufl.TestFunction(V)

	a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
	L = f * v * ufl.dx
	problem = LinearProblem(a, L,
				bcs = [bc],
				petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
	
	uh = problem.solve()

	return V, uh

def Energy(u):
	# return the energy of the approximate solution
	energy = .5*ufl.dot(ufl.grad(u), ufl.grad(u))*ufl.dx
	return fem.assemble_scalar(fem.form(energy))

def GetGradType0(domain, uh):
	W = fem.VectorFunctionSpace(domain, ("DG", 0))
	Guh = fem.Function(W)
	G_exp = fem.Expression(ufl.grad(uh), W.element.interpolation_points())
	Guh.interpolate(G_exp)
	return W, Guh

def GetGradType1(domain, uh):
	W = fem.VectorFunctionSpace(domain, ("Lagrange", 1))
	w = ufl.TrialFunction(W)
	v = ufl.TestFunction(W)

	a = ufl.inner(w, v) * ufl.dx
	L = ufl.inner(ufl.grad(uh), v) * ufl.dx
	problem = LinearProblem(a, L)
	Guh = problem.solve()
	return W, Guh

def PlotSolution(domain, V, uh):
	p = pyvista.Plotter()
	plot.vtk_mesh(V)
	grid = pyvista.UnstructuredGrid(*vtk_mesh(V))
	grid.point_data["u"] = uh.x.array
	grid.set_active_scalars("u")
	warp = grid.warp_by_scalar("u", factor=1)
	actor = p.add_mesh(warp, show_edges=True)
	p.show_axes()
	p.show()

def PlotGrad0(domain, V, uh):
	W, Guh = GetGradType0(domain, uh)
	p = pyvista.Plotter()
	grid = pyvista.UnstructuredGrid(*vtk_mesh(V))

	# We include ghosts cells as we access all degrees of freedom (including ghosts) on each process
	top_imap = domain.topology.index_map(domain.topology.dim)
	num_cells = top_imap.size_local + top_imap.num_ghosts
	midpoints = mesh.compute_midpoints(domain, domain.topology.dim, range(num_cells))

	num_dofs = W.dofmap.index_map.size_local + W.dofmap.index_map.num_ghosts
	assert (num_cells == num_dofs)
	values = np.zeros((num_dofs, 3), dtype=np.float64)
	values[:, :domain.geometry.dim] = Guh.x.array.real.reshape(num_dofs, W.dofmap.index_map_bs)
	cloud = pyvista.PolyData(midpoints)
	cloud["Gu"] = values
	glyphs = cloud.glyph("Gu", factor=0.05)
	actor = p.add_mesh(grid, style="wireframe", color="k")
	actor2 = p.add_mesh(glyphs)
	p.show_axes()
	p.show()

def PlotGrad1(domain, V, uh):
	W, Guh = GetGradType1(domain, uh)
	p = pyvista.Plotter()
	topology, cell_types, geometry = vtk_mesh(W)
	grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
	grid["Gu"] = Guh.x.array.reshape((geometry.shape[0], 3))
	actor_0 = p.add_mesh(grid, style="wireframe", color="k")
	warped = grid.warp_by_vector("Gu", factor=0.05)
	actor_1 = p.add_mesh(warped, show_edges=True)
	p.show_axes()
	p.show()

def TagAndFaces(domain, A, b, K):
	#print(A, b, K)
	# Each fk function must accept the set of all nodes x
	# and return true or false for each element of x 
	x = ufl.SpatialCoordinate(domain)
	faces = []
	for k in range(K):
		fk = lambda x, j=k: np.isclose(x[0]*A[j, 0] + x[1]*A[j, 1] + x[2]*A[j, 2] - b[j], 0.)
		faces.append((k+1, fk))
	return faces

def MarkedFaces(domain, faces):
	facet_indices, facet_markers = [], []
	fdim = domain.topology.dim - 1
	
	for (marker, locator) in faces:
		facets = mesh.locate_entities(domain, fdim, locator)
		facet_indices.append(facets)
		facet_markers.append(np.full_like(facets, marker))
	facet_indices = np.hstack(facet_indices).astype(np.int32)
	facet_markers = np.hstack(facet_markers).astype(np.int32)
	sorted_facets = np.argsort(facet_indices)
	facet_tag = mesh.meshtags(domain, fdim,
			facet_indices[sorted_facets],
			facet_markers[sorted_facets])

	# depuration
	#domain.topology.create_connectivity(domain.topology.dim-1, domain.topology.dim)
	#with XDMFFile(domain.comm, "facet_tags.xdmf", "w") as xdmf:
	#	xdmf.write_mesh(domain)
	#	xdmf.write_meshtags(facet_tag, domain.geometry)

	return facet_tag

def dsXFace(domain, K, A, b):
	faces = TagAndFaces(domain, A, b, K)
	facet_tag = MarkedFaces(domain, faces)
	return ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)

def IniMul(K, uh, n, ds, uno):
	d = np.zeros(K)
	f = np.zeros(K)
	for k in range(K):
		int_k = (ufl.dot(ufl.grad(uh), n)**2) * ds(k+1)
		d[k] = fem.assemble_scalar(fem.form(int_k))
		int_k = uno * ds(k+1)
		f[k] = fem.assemble_scalar(fem.form(int_k))
	return np.max(d/f)

def DuXn2(K, uh, n, ds, multiplier):
	d = np.zeros(K)
	for k in range(K):
		int_k = (multiplier - ufl.dot(ufl.grad(uh), n)**2) * ds(k+1)
		d[k] = fem.assemble_scalar(fem.form(int_k))
	return d

def DuXn2_xXv(K, uh, x, n, ds, multiplier, v):
	d = np.zeros(K)
	for k in range(K):
		int_k = ufl.dot(x, v[k]) * (multiplier - ufl.dot(ufl.grad(uh), n)**2) * ds(k+1)
		d[k] = fem.assemble_scalar(fem.form(int_k))
	return d
	
def fkds(K, f, n, ds):
	d = np.zeros(K)
	for k in range(K):
		d[k] = fem.assemble_scalar(fem.form(f * ds(k+1)))
	return d
	
def fkXvds(K, f, x, n, ds, v):
	d = np.zeros(K)
	for k in range(K):
		d[k] = fem.assemble_scalar(fem.form(ufl.dot(x, v[k]) * f * ds(k+1)))
	return d

def Constant(domain, value):
	return fem.Constant(domain, default_scalar_type(value))
	
def GetVal(formula):
	return fem.assemble_scalar(fem.form(formula))
	
def AllDerivatives1(K, u, x, n, ds, vi, vj, e1Fx):
	d = np.zeros((6, K))
	for k in range(K):
		xXvi = ufl.dot(x, vi[k])
		xXvj = ufl.dot(x, vj[k])
		d[0, k] = GetVal(ufl.dot(ufl.grad(u), n)**2 * ds(k+1))
		d[1, k] = GetVal(xXvi * ufl.dot(ufl.grad(u), n)**2 * ds(k+1))
		d[2, k] = GetVal(xXvj * ufl.dot(ufl.grad(u), n)**2 * ds(k+1))
		d[3, k] = GetVal(xXvi * ds(k+1))
		d[4, k] = GetVal(xXvj * ds(k+1))
		d[5, k] = GetVal(e1Fx * ds(k+1))
	return d

def AllDerivatives2(K, u, x, N, ds, vi, vj, e1Fx):
	d = np.zeros((6, K))
	for k in range(K):
		xXvi = ufl.dot(x, vi[k])
		xXvj = ufl.dot(x, vj[k])
		d[0, k] = GetVal(ufl.dot(ufl.grad(u), N[k])**2 * ds(k+1))
		d[1, k] = GetVal(xXvi * ufl.dot(ufl.grad(u), N[k])**2 * ds(k+1))
		d[2, k] = GetVal(xXvj * ufl.dot(ufl.grad(u), N[k])**2 * ds(k+1))
		d[3, k] = GetVal(xXvi * ds(k+1))
		d[4, k] = GetVal(xXvj * ds(k+1))
		d[5, k] = GetVal(e1Fx * ds(k+1))
	return d

def CartesianToSpherical(K, X):
	Alpha = np.zeros(K)
	Beta = np.zeros(K)
	r0 = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
	Alpha = np.arctan2(X[:, 1], X[:, 0])
	Beta = np.arctan2(r0, X[:,2])
	return Alpha, Beta

def SphericalToCartesian(K, Alpha, Beta, nde = 14):
	X = np.zeros((K, 3))
	X[:, 0] = np.cos(Alpha)*np.sin(Beta)
	X[:, 1] = np.sin(Alpha)*np.sin(Beta)
	X[:, 2] = np.cos(Beta)
	return X

def Vectors_i_and_j(K, Alpha, Beta, nde = 14):
	sA = np.sin(Alpha)
	cA = np.cos(Alpha)
	sB = np.sin(Beta)
	cB = np.cos(Beta)

	vi = np.zeros((K, 3))
	vi[:, 0] = -sA*sB
	vi[:, 1] = cA*sB

	vj = np.zeros((K, 3))
	vj[:, 0] = cA*cB
	vj[:, 1] = sA*cB
	vj[:, 2] = -sB

	return vi, vj

def GetVijFx(K, domain, Alpha, Beta, nde = 14):
	vi, vj = Vectors_i_and_j(K, Alpha, Beta, nde)
	Vi = [ufl.as_vector([Constant(domain, vi[k][l]) for l in range(3)]) for k in range(K)]
	Vj = [ufl.as_vector([Constant(domain, vj[k][l]) for l in range(3)]) for k in range(K)]
	return Vi, Vj
	

def Derivatives(uh, ds, K, domain, vi, vj):
	DAlpha = 0.5*DuXn2_xXv(uh, ds, K, domain, vi)
	DBeta = 0.5*DuXn2_xXv(uh, ds, K, domain, vj)
	DLambda = -0.5*DuXn2(uh, ds, K)
	return DAlpha, DBeta, DLambda

def ValueJ(Alpha, Beta, Lambda):
	A = SphericalToCartesian(8, Alpha, Beta)
	b = Lambda
	domain = GetDomain(8, A, b, size_mesh = 0.25)
	V, uh = SolvePDE01(domain)
	energy = .5*ufl.dot(ufl.grad(uh), ufl.grad(uh))*ufl.dx
	return fem.assemble_scalar(fem.form(energy))

def InterativeMethod(A, B, L, model):
	steps = 3
	t = 0.5
	DA, DB, DL, VJ = model(A, B, L)
	for stp in range(steps):
		Anew = A + t*DA
		Bnew = B + t*DB
		Lnew = L + t*DL
		if ValueJ(Anew, Bnew, Lnew) < VJ: break
		t = t**2 # modificar

	return Anew, Bnew, Lnew

def Run(A, B, L, ModFunc, steps):
	for i in range(steps):
		A, B, L = InterativeMethod(A, B, L, ModFunc)
	return A, B, L

def GetVertices(A, b, vertices):
	v = []
	for idxs in vertices:
		v.append(np.linalg.solve(A[idxs], b[idxs]))
	return np.array(v)

def Polyedral01():
	# number of planes
	K = 8
	# Ax <= b
	A = np.array([[-1, 0, 0],
			[0, -1, 0],
			[0, 0, -1],
			[1, 1, 1],
			[0, 0, 1],
			[0, 1, 0],
			[1, 0, 0],
			[-1,-1/2,1]])
	b = np.array([1, 1, 1, 1, 1, 0.8, 0.6, 2])
	A, b = Normalize(A, b)
	return K, A, b

def Polyedral02():
	# number of planes
	K = 6
	# Ax <= b
	A = np.array([[-1., -0.2, 0.],
				[0, -1., 0.],
				[0, 0., -1.],
				[0., 0., 1.],
				[0., 1., 0.],
				[1., 0.5, 0.]])
	b = np.array([1. , 1. , 1., 1., 1., 1.])
	A, b = Normalize(A, b)
	return K, A, b
	
def Polyedral03():
	# number of planes
	K = 7
	# Ax <= b
	A = np.array([[-1., 0.5, 0.],
				[-1., -0.5, 0.],
				[0, -1., 0.],
				[0, 0., -1.],
				[0., 0., 1.],
				[0., 1., 0.],
				[1., 0, 0.]])
	b = np.array([1., 1. , 1. , 1., 1., 1., 1.])
	A, b = Normalize(A, b)
	return K, A, b
	
def Polyedral04():
	# number of planes
	K = 8
	# Ax <= b
	A = np.array([[-1., 0.5, 0.],
				[-1., -0.5, 0.],
				[0, -1., 0.],
				[0, 0., -1.],
				[0., 0., 1.],
				[0., 1., 0.],
				[1., 0.5, 0.],
				[1., -0.5, 0.]])
	b = np.array([1., 1., 1. , 1.5 , 1.5, 1., 1., 1.])
	A, b = Normalize(A, b)
	# vertices
	return K, A, b

def ModFunc01(Alpha, Beta, Lambda):
	K = 8
	A = SphericalToCartesian(K, Alpha, Beta)
	b = Lambda
	domain = GetDomain(K, A, b, size_mesh = 0.25)
	V, uh = SolvePDE01(domain)
	ds = dsXFace(domain, K, A, b)
	VJ = ValueJ(Alpha, Beta, Lambda)
	vi, vj = Vectors_i_and_j(K, Alpha, Beta)
	DA, DB, DL = Derivatives(uh, ds, K, domain, vi, vj)
	return DA, DB, DL, VJ

def Example00():
	
	size_mesh = 0.25
	K, A, b = Polyedral01()
	domain = GetDomain(K, A, b, size_mesh)

	V, uh = SolvePDE01(domain)

	print('Energy = ', Energy(uh))

	#PlotSolution(domain, V, uh)
	#PlotGrad0(domain, V, uh)
	#PlotGrad1(domain, V, uh)
	
	ds = dsXFace(domain, K, A, b)
	v = np.random.rand(K, 3)
	d = DuXn2(uh, ds, K)
	d = DuXn2_xXv(uh, ds, K, domain, v)
	print(d)
	
def IterMethod(model, steps, t, factor):
	# Gradiend Descent Method
	cost = model.cost
	A, B, L, m, DA, DB, DL, Dm = model.Get()
	for j in range(steps):
		model.Set(A + t*DA, B + t*DB, L + t*(1E-2)*DL, m + t*Dm)
		if model.cost < cost : break
		t = factor*t
	return cost, j+1
	
def IterMethodVolume(model, steps, t, factor):
	# Gradiend Descent Method
	cost = model.cost
	A, B, L, DA, DB, DL = model.Get()
	for j in range(steps):
		model.Set(A + t*DA, B + t*DB, L + t*(1E-2)*DL)
		if model.cost < cost : break
		t = factor*t
	return j+1

def PrintIteration(i, cost, effort, volume, h, m, J, mxcos, var):
	# Print iteration, normalized cost rate
	si = "{:06.0f}".format(i) # iteration
	sc = "{:07.4f}".format(cost) # cost
	sv = "{:07.4f}".format(volume) # volume
	sh = "{:06.2f}".format(h) # size mesh
	sm = "{:08.4f}".format(m) # multiplier
	sJ = "{:010.6f}".format(J) # J
	sx = "{:06.4f}".format(mxcos) # maximum cosine
	svr = "{:010.8f}".format(var)
	print(f"i = {si}, Cost = {sc}, Effort = {effort}, Vol = {sv}, h = {sh}, m = {sm}, J = {sJ}, cos = {sx}, var = {svr}")

		
def NormalizeDer(DA, DB, DL):
	n = np.linalg.norm(np.hstack((DA, DB, DL)), ord = 2)
	return DA/n, DB/n, DL/n
	
def GetSizeMesh(domain):
	return (VolumeFx(domain)**(1./3.))/6.
	
def CheckVertices(v, nv):
	# This function calculates the cosine of every
	# pair of vertices and returns the maximum cosine.
	u = []
	for i in range(nv):
		u.append(v[i]/np.linalg.norm(v[i], ord = 2))
	d = []
	for i in range(nv - 1):
		for j in range(i + 1, nv):
			d.append(np.inner(u[i], u[j]))
	return max(d)
	
class Save():
	
	def __init__(self, mdl, add = ""):
		self.filename = mdl.name + add
		self.mdl = mdl
		self.coordinate1 = []
		self.coordinate2 = []
		self.indicators1 = []
		self.indicators2 = []
	
	def Add1(self):
		x = np.hstack([self.mdl.Alph, self.mdl.Beta, self.mdl.Lamb])
		dx = np.hstack([self.mdl.DAlph, self.mdl.DBeta, self.mdl.DLamb])
		ndx = np.linalg.norm(dx, ord = 2)
		row = [-self.mdl.J + self.mdl.par_l*self.mdl.volume, \
			   self.mdl.J, self.mdl.par_l, self.mdl.volume, ndx]
		self.coordinate1.append(x)
		self.indicators1.append(row)
	
	def Add2(self, lagr_cost, rho):
		x = np.hstack([self.mdl.Alph, self.mdl.Beta, self.mdl.Lamb])
		nx = np.linalg.norm(x, ord = 2)
		newLamb = self.mdl.Lamb + self.mdl.DLamb
		Px = np.hstack([self.mdl.Alph + self.mdl.DAlph, \
						self.mdl.Beta + self.mdl.DBeta, \
						np.where(newLamb < 0.1, 0.1, newLamb)])
		gap = np.linalg.norm(x - Px, ord = 2)
		row = [lagr_cost, nx, \
			   abs(self.mdl.par_l), abs(self.mdl.par_u), abs(self.mdl.par_z), \
			   gap, (1./rho)*abs(self.mdl.par_l - self.mdl.par_u), \
			   self.mdl.J, self.mdl.volume]
		self.coordinate2.append(x)
		self.indicators2.append(row)
		
	def SaveFile(self):
		np.save(self.filename + '_coor1', np.array(self.coordinate1))
		np.save(self.filename + '_indi1', np.array(self.indicators1))
		np.save(self.filename + '_coor2', np.array(self.coordinate2))
		np.save(self.filename + '_indi2', np.array(self.indicators2))
	
class SaintVenant2():
	
	def __init__(self, K, A, b, N, nbrv, idxv):
		self.name = "SaintVenant"
		self.K = K
		self.A = A
		self.b = b
		self.N = N
		self.nbrv = nbrv
		self.idxv = idxv
		# primary calculations
		self.vertices = GetVertices(self.A, self.b, self.idxv)
		self.volume = ConvexHull(self.vertices).volume
		self.h = (self.volume**(1./3.))/self.N
		# secondary calculations
		self.var = np.var(np.linalg.norm(self.vertices, ord = 2, axis = 1))
		self.mxcos = CheckVertices(self.vertices, self.nbrv)
		# approximate solution
		self.domain = GetDomain(self.K, self.A, self.b, self.h, True)
		self.uh = None
		self.PDEquation()
		# cost functional
		self.J = fem.assemble_scalar(fem.form(self.uh * ufl.dx))
		# (x, z, lambda, mu) and delta
		self.Alph, self.Beta = CartesianToSpherical(self.K, self.A)
		self.Lamb = self.b
		self.par_z = 0.
		self.par_l = 0.
		self.par_u = 0.
		self.par_d = 0.5
		# derivative
		self.DAlph = None
		self.DBeta = None
		self.DLamb = None
		self.Derivatives()
		self.fixed_volume = np.pi
		
	def Calculate(self):
		self.A = SphericalToCartesian(self.K, self.Alph, self.Beta)
		self.b = self.Lamb
		# primary calculations
		self.vertices = GetVertices(self.A, self.b, self.idxv)
		self.volume = ConvexHull(self.vertices).volume
		self.h = (self.volume**(1./3.))/self.N
		# secondary calculations
		self.var = np.var(np.linalg.norm(self.vertices, ord = 2, axis = 1))
		self.mxcos = CheckVertices(self.vertices, self.nbrv)
		# approximate solution
		self.domain = GetDomain(self.K, self.A, self.b, self.h, False)
		self.PDEquation()
		self.J = fem.assemble_scalar(fem.form(self.uh * ufl.dx))
		# derivative
		self.Derivatives()

	def PDEquation(self):
		V = fem.FunctionSpace(self.domain, ("Lagrange", 1))
		f = Constant(self.domain, 1.)
		u = ufl.TrialFunction(V)
		v = ufl.TestFunction(V)
		a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
		L = f * v * ufl.dx
		uD = fem.Function(V)
		uD.interpolate(lambda x: 0.*x[0])
		self.domain.topology.create_connectivity(2, 3)
		boundary_facets = mesh.exterior_facet_indices(self.domain.topology)
		boundary_dofs = fem.locate_dofs_topological(V, 2, boundary_facets)
		bc = fem.dirichletbc(uD, boundary_dofs)
		problem = LinearProblem(a, L, bcs = [bc],
					petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
		self.uh = problem.solve()
		
	def Derivatives(self):
		n = ufl.FacetNormal(self.domain)
		ds = dsXFace(self.domain, self.K, self.A, self.b)
		vi, vj = GetVijFx(self.K, self.domain, self.Alph, self.Beta)
		x = ufl.SpatialCoordinate(self.domain)
		ufl_mult = Constant(self.domain, self.par_l)
		self.DAlph = DuXn2_xXv(self.K, self.uh, x, n, ds, ufl_mult, vi)
		self.DBeta = DuXn2_xXv(self.K, self.uh, x, n, ds, ufl_mult, vj)
		self.DLamb = -DuXn2(self.K, self.uh, n, ds, ufl_mult)
		
	def Lagrangian(self, par_alp, par_bet):
		return -self.J + \
			   self.par_l*(self.volume - self.fixed_volume - self.par_z) + \
			   self.par_u*self.par_z + \
			   (par_alp/2.)*self.par_z**2 - \
			   (par_bet/2.)*(self.par_l - self.par_u)**2 


def Example04():
	# pentagonal_prism
	# hexagonal_prism
	# tetrahedron
	# cube
	# dodecahedron 
	# chamfered_tetrahedron
	pl = dodecahedron
	K = pl["K"]
	A = np.array(pl["A"])
	b = np.array(pl["b"])
	A, b = Normalize(A, b)
	
	mdl = SaintVenant2(K, A, b, 10., pl["nv"], pl["vertices"])
	sf = Save(mdl)
	
	print("A:")
	print(mdl.A)
	print("b:")
	print(mdl.b[:, None])
	print("Vertex norms:")
	
	for v in mdl.vertices:
		print("{:07.4f}".format(np.linalg.norm(np.array(v), ord = 2)))
	
	par_alp = 2000. # penalty parameter
	par_bet = 0.5 # proximal parameter
	par_eta = 500. # eta is large enough
	par_rho = par_alp/(1. + par_alp*par_bet)
	par_r = 0.9
	
	t = 0.01
	factor = 0.8
	steps = 4
	mdl.par_l = 1.
	
	print('Stage 1')
	PrintIteration(0, -mdl.J + mdl.par_l*mdl.volume, 0, mdl.volume, mdl.h, mdl.par_l, mdl.J, mdl.mxcos, mdl.var)
	sf.Add1()
	
	for i in range(100):
		
		pre_cost = -mdl.J + mdl.par_l*mdl.volume
		pre_var = mdl.var
		
		for j in range(steps):
			
			mdl.Alph = mdl.Alph + t*mdl.DAlph
			mdl.Beta = mdl.Beta + t*mdl.DBeta
			mdl.Lamb = mdl.Lamb + t*(1E-2)*mdl.DLamb
			
			mdl.Calculate()
			
			if -mdl.J + mdl.par_l*mdl.volume < pre_cost : break
			t = factor*t
			
		if mdl.mxcos > 0.995 : break
		if pre_var < mdl.var : break	
		
		PrintIteration(i+1, -mdl.J + mdl.par_l*mdl.volume, j+1, mdl.volume, mdl.h, mdl.par_l, mdl.J, mdl.mxcos, mdl.var)
	
	mdl.fixed_volume = mdl.volume
	mdl.par_l = 1.
	
	print('Stage 2')
	cost = mdl.Lagrangian(par_alp, par_bet)
	PrintIteration(0, cost, 0, mdl.volume, mdl.h, mdl.par_l, mdl.J, mdl.mxcos, mdl.var)
	sf.Add2(cost, par_r)
	
	for i in range(100):
		
		mdl.Alph = mdl.Alph + mdl.DAlph/par_eta
		mdl.Beta = mdl.Beta + mdl.DBeta/par_eta
		newLamb = mdl.Lamb + mdl.DLamb/par_eta
		mdl.Lamb = np.where(newLamb < 0.1, 0.1, newLamb)
		
		mdl.Calculate()
		
		if mdl.mxcos > 0.995 : break
		
		ftr = mdl.par_d/((mdl.par_l - mdl.par_u)**2 + 1.)
		
		mdl.par_u = mdl.par_u + ftr*(mdl.par_l - mdl.par_u) 
		mdl.par_l = mdl.par_u + par_rho*(mdl.volume - mdl.fixed_volume)
		
		mdl.par_z = (mdl.par_l - mdl.par_u)/par_alp
		mdl.par_d = par_r*mdl.par_d
		
		cost = mdl.Lagrangian(par_alp, par_bet)
		
		sf.Add2(cost, par_r)
		PrintIteration(i+1, cost, 0, mdl.volume, mdl.h, mdl.par_l, mdl.J, mdl.mxcos, mdl.var)
			

	print("Vertex norms:")
	print(np.linalg.norm(mdl.vertices, ord = 2, axis = 1)[:, None])
	print("A:")
	print(mdl.A)
	print("b:")
	print(mdl.b[:, None])
	
	
	sf.SaveFile()
	s = GetDomain(mdl.K, mdl.A, mdl.b, mdl.h, True)

def ReadFileDodecahedron():
	colors = np.random.rand(20)
	
	with open('points.npy', 'rb') as f:
		for i in range(0, 8001, 200):
			v = np.load(f)
			print(v)
			fig = plt.figure(figsize=(6, 6))
			plt.scatter(v[:, 0], v[:, 1], c = colors)
			plt.xlim([0-0.2, 2+0.2])
			plt.ylim([0-0.2, 2+0.2])
			plt.grid()
			fig.tight_layout()
			plt.savefig('img'+str(i)+'.png')
			

	
class Volume():
	def __init__(self, K, A, b, N, vts):
		self.K = K
		self.A = A
		self.b = b
		self.N = N
		self.vts = vts
		self.vertices = GetVertices(self.A, self.b, self.vts)
		self.volume = ConvexHull(self.vertices).volume
		self.h = (self.volume**(1./3.))/self.N
		self.mxcos = CheckVertices(self.vertices)
		# PDE equation variables
		self.domain = GetDomain(self.K, self.A, self.b, self.h,True)
		self.V = fem.FunctionSpace(self.domain, ("Lagrange", 1))
		self.cost = None
		self.Alph, self.Beta = CartesianToSpherical(self.K, self.A)
		self.Lamb = self.b
		self.DAlph = None
		self.DBeta = None
		self.DLamb = None
		self.f = None
		self.Derivatives()
		self.CostFunctional()
		
	def Get(self):
		self.Derivatives()
		return self.Alph, self.Beta, self.Lamb, \
				self.DAlph, self.DBeta, self.DLamb
				
	def Set(self, *args):
		self.Alph, self.Beta, self.Lamb = args
		self.A = SphericalToCartesian(self.K, self.Alph, self.Beta)
		self.b = self.Lamb
		self.vertices = GetVertices(self.A, self.b, self.vts)
		self.mxcos = CheckVertices(self.vertices)
		self.volume = ConvexHull(self.vertices).volume
		self.h = (self.volume**(1./3.))/self.N
		self.domain = GetDomain(self.K, self.A, self.b, self.h, False)
		self.Derivatives()
		self.CostFunctional()
		
	def CostFunctional(self):
		self.cost = fem.assemble_scalar(fem.form(self.f * ufl.dx))
		
	def Derivatives(self):
		self.V = fem.FunctionSpace(self.domain, ("Lagrange", 1))
		self.f = fem.Function(self.V)
		self.f.interpolate(lambda x: (x[0]/2.)**2 + (x[1]/1.)**2 + (x[2]/0.5)**2 - 1.)
		n = ufl.FacetNormal(self.domain)
		ds = dsXFace(self.domain, self.K, self.A, self.b)
		vi, vj = GetVijFx(self.K, self.domain, self.Alph, self.Beta)
		x = ufl.SpatialCoordinate(self.domain)
		self.DAlph = fkXvds(self.K, self.f, x, n, ds, vi)
		self.DBeta = fkXvds(self.K, self.f, x, n, ds, vj)
		self.DLamb = -fkds(self.K, self.f, n, ds)

class SaintVenant():
	
	def __init__(self, K, A, b, N, nv, vts):
		self.K = K
		self.A = A
		self.b = b
		self.N = N
		self.vts = vts
		self.nv = nv
		self.vertices = GetVertices(self.A, self.b, self.vts)
		self.volume = ConvexHull(self.vertices).volume
		self.h = (self.volume**(1./3.))/self.N
		self.var = np.var(np.linalg.norm(self.vertices, ord = 2, axis = 1))
		self.mxcos = CheckVertices(self.vertices, self.nv)
		# PDE equation variables
		self.domain = GetDomain(self.K, self.A, self.b, self.h,True)
		self.uh = None
		self.V = None
		self.PDEquation()
		self.cost = None
		self.J = None
		# alternative:
		# self.volume = VolumeFx(self.domain)
		# Parametrization variables
		self.Alph, self.Beta = CartesianToSpherical(self.K, self.A)
		self.Lamb = self.b
		self.mult = 1.
		# alternative:
		# n = ufl.FacetNormal(self.domain)
		# ds = dsXFace(self.domain, self.K, self.A, self.b)
		# self.mult = fem.assemble_scalar(fem.form(self.uh * ufl.dx))/VolumeFx(self.domain)
		self.DAlph = None
		self.DBeta = None
		self.DLamb = None
		self.Dmult = None 
		self.Derivatives()
		self.CostFunctional()
	
	def Get(self):
		self.Derivatives()
		return self.Alph, self.Beta, self.Lamb, self.mult, \
				self.DAlph, self.DBeta, self.DLamb, self.Dmult

	def Set(self, *args):
		self.Alph, self.Beta, self.Lamb, self.mult = args
		self.mult = 1.
		self.A = SphericalToCartesian(self.K, self.Alph, self.Beta)
		self.b = self.Lamb
		self.vertices = GetVertices(self.A, self.b, self.vts)
		self.var = np.var(np.linalg.norm(self.vertices, ord = 2, axis = 1))
		self.mxcos = CheckVertices(self.vertices, self.nv)
		self.volume = ConvexHull(self.vertices).volume
		self.h = (self.volume**(1./3.))/self.N
		self.domain = GetDomain(self.K, self.A, self.b, self.h, False)
		# alternative
		# self.volume = VolumeFx(self.domain)
		self.PDEquation()
		self.Derivatives()
		self.CostFunctional()

	def PDEquation(self):
		self.V = fem.FunctionSpace(self.domain, ("Lagrange", 1))
		f = Constant(self.domain, 1.)
		u = ufl.TrialFunction(self.V)
		v = ufl.TestFunction(self.V)
		a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
		L = f * v * ufl.dx
		#bc = GetuDValue(self.domain, self.V, 0.) # boundary value
		uD = fem.Function(self.V)
		uD.interpolate(lambda x: 0.*x[0])
		self.domain.topology.create_connectivity(2, 3)
		boundary_facets = mesh.exterior_facet_indices(self.domain.topology)
		boundary_dofs = fem.locate_dofs_topological(self.V, 2, boundary_facets)
		bc = fem.dirichletbc(uD, boundary_dofs)
		problem = LinearProblem(a, L, bcs = [bc],
					petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
		self.uh = problem.solve()
	
	def CostFunctional(self):
		self.J = fem.assemble_scalar(fem.form(self.uh * ufl.dx))
		self.cost = -self.J + self.mult*self.volume
		
	def Derivatives(self):
		n = ufl.FacetNormal(self.domain)
		ds = dsXFace(self.domain, self.K, self.A, self.b)
		vi, vj = GetVijFx(self.K, self.domain, self.Alph, self.Beta)
		x = ufl.SpatialCoordinate(self.domain)
		ufl_mult = Constant(self.domain, self.mult)
		self.DAlph = DuXn2_xXv(self.K, self.uh, x, n, ds, ufl_mult, vi)
		self.DBeta = DuXn2_xXv(self.K, self.uh, x, n, ds, ufl_mult, vj)
		self.DLamb = -DuXn2(self.K, self.uh, n, ds, ufl_mult)
		self.Dmult = 0.# (VolumeFx(self.domain)-np.pi)


def Example01():
	K, A, Lambda = Polyedral01()
	Alpha, Beta = CartesianToSpherical(K, A)
	AlphaNew, BetaNew, LambdaNew = Run(Alpha, Beta, Lambda, ModFunc01, steps = 10)
	print(ValueJ(Alpha, Beta, Lambda))
	print(ValueJ(AlphaNew, BetaNew, LambdaNew))
	A = SphericalToCartesian(K, AlphaNew, BetaNew)
	b = LambdaNew
	domain = GetDomain(K, A, b, size_mesh = 0.2)

def Example02():
	# pentagonal_prism
	# hexagonal_prism
	# tetrahedron
	# cube
	# dodecahedron
	# chamfered_tetrahedron
	pl = cube
	K = pl["K"]
	A = np.array(pl["A"])
	b = np.array(pl["b"])
	A, b = Normalize(A, b)
	
	mdl = SaintVenant(K, A, b, 10., pl["nv"], pl["vertices"])
	
	print("A:")
	print(mdl.A)
	print("b:")
	print(mdl.b[:, None])
	print("Vertex norms:")
	# alternative:
	# vtx = compute_polytope_vertices(mdl.A.copy(), mdl.b.copy())
	for v in mdl.vertices:
		print("{:07.4f}".format(np.linalg.norm(np.array(v), ord = 2)))
		
	PrintIteration(0, mdl.cost, 0, mdl.volume, mdl.h, mdl.mult, mdl.J, mdl.mxcos, mdl.var)
	
	for i in range(100):
		previous_var = mdl.var
		previous_cost, effort = IterMethod(mdl, steps = 4, t = 0.01, factor = 0.8)
		if mdl.mxcos > 0.995 : break
		if previous_var < mdl.var : break
		PrintIteration(i+1, mdl.cost, effort, mdl.volume, mdl.h, mdl.mult, mdl.J, mdl.mxcos, mdl.var)
	
	print("Vertex norms:")
	# alternative:
	# vtx = compute_polytope_vertices(mdl.A.copy(), mdl.b.copy())
	vts = GetVertices(mdl.A, mdl.b, pl["vertices"])
	for v in vts:
		print("{:07.4f}".format(np.linalg.norm(np.array(v), ord = 2)))
	
	print("A:")
	print(mdl.A)
	print("b:")
	print(mdl.b[:, None])
	
	# plot boundary grid
	#p = pyvista.Plotter()
	#grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(mdl.V))
	#p.add_mesh(grid, style="wireframe", color="k")
	#p.show_axes()
	#p.show()
	
	s = GetDomain(mdl.K, mdl.A, mdl.b, mdl.h, True)
	
def Example03():
	# pentagonal_prism
	# hexagonal_prism
	# tetrahedron
	# cube
	# dodecahedron
	# chamfered_tetrahedron
	pl = cube
	K = pl["K"]
	A = np.array(pl["A"])
	b = np.array(pl["b"])
	A, b = Normalize(A, b)
	
	mdl = Volume(K, A, b, 6., pl["vertices"])
	
	print("A:")
	print(mdl.A)
	print("b:")
	print(mdl.b[:, None])
	print("Vertices:")
	for v in mdl.vertices:
		print(v)
		print("{:07.4f}".format(np.linalg.norm(np.array(v), ord = 2)))
		
	PrintIteration(0, mdl.cost, 0, mdl.volume, mdl.h, 0., 0., mdl.mxcos, 0.)
	
	for i in range(2000):
		effort = IterMethodVolume(mdl, steps = 4, t = 0.01, factor = 0.8)
		if mdl.mxcos > 0.995 : break
		PrintIteration(i+1, mdl.cost, effort, mdl.volume, mdl.h, 0., 0., mdl.mxcos, 0.)
	
	print("A:")
	print(mdl.A)
	print("b:")
	print(mdl.b[:, None])
	print("Vertices:")
	for v in mdl.vertices:
		print(v)
		print("{:07.4f}".format(np.linalg.norm(np.array(v), ord = 2)))
	
	s = GetDomain(mdl.K, mdl.A, mdl.b, mdl.h, True)

	
def ModelforTest(domain):
	V = fem.FunctionSpace(domain, ("Lagrange", 1))
	f = fem.Constant(domain, default_scalar_type(-6.))
	u = ufl.TrialFunction(V)
	v = ufl.TestFunction(V)
	a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
	L = f * v * ufl.dx
	uD = fem.Function(V)
	uD.interpolate(lambda x: 1 + x[0]**2 + x[1]**2 + x[2]**2)
	domain.topology.create_connectivity(2, 3)
	boundary_facets = mesh.exterior_facet_indices(domain.topology)
	boundary_dofs = fem.locate_dofs_topological(V, 2, boundary_facets)
	bc = fem.dirichletbc(uD, boundary_dofs)
	problem = LinearProblem(a, L, bcs = [bc],
		petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
	uh = problem.solve()
	V2 = fem.FunctionSpace(domain, ("Lagrange", 2))
	uex = fem.Function(V2)
	uex.interpolate(lambda x: 1 + x[0]**2 + x[1]**2 + x[2]**2)
	L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
	error_local = fem.assemble_scalar(L2_error)
	error_L2 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
	error_max = np.max(np.abs(uD.x.array-uh.x.array))
	return uh, error_L2, error_max

def Test():
	# Test
	print('TEST\n')
	K, A, b = Polyedral02()
	print('Matrix A:')
	print(A)
	Alph, Beta = CartesianToSpherical(K, A)
	Lamb = b
	A = SphericalToCartesian(K, Alph, Beta)
	print('Matrix A (after conversion):')
	print(A)
	domain = GetDomain(K, A, b, 0.1, False)
	print('Volumen: ', VolumeSc(A, b))
	print('Volumen: ', VolumeFx(domain))
	# approximate solution
	uh, error_L2, error_max, = ModelforTest(domain)
	print(f"L2 error: {error_L2}, Max error: {error_max}")
	# Normal vectors
	N = []
	for k in range(K):
		n0 = Constant(domain, A[k, 0])
		n1 = Constant(domain, A[k, 1])
		n2 = Constant(domain, A[k, 2])
		N.append(ufl.as_vector([n0, n1, n2]))
	n = ufl.FacetNormal(domain)
	# ds for boundary integration
	ds = dsXFace(domain, K, A, b)
	x = ufl.SpatialCoordinate(domain)
	Vi, Vj = GetVijFx(K, domain, Alph, Beta)
	e1Fx = Constant(domain, 1.)
	# derivatives
	d1 = AllDerivatives1(K, uh, x, n, ds, Vi, Vj, e1Fx)
	d2 = AllDerivatives2(K, uh, x, N, ds, Vi, Vj, e1Fx)
	print('Derivatives:')
	print(np.round(d1, 2))
	print(np.round(d2, 2))
	print('Difference:')
	print(np.round(d1-d2, 4))
	# Derivatives
	dAlph = d1[1]
	dBeta = d1[2]
	dLamb = -d1[0]
	wA, wB, wL = 1., 1., 1E-4
	norm = np.sum(dAlph*dAlph) + np.sum(dBeta*dBeta) + np.sum(dLamb*dLamb)
	print(f"Norm: {norm}")
	
	for t in [0.08, 0.06, 0.04, 0.02, 0.005, 0.000001]:
		newAlph = Alph + t*wA*dAlph
		newBeta = Beta + t*wB*dBeta
		newLamb = Lamb + t*wL*dLamb
		newA = SphericalToCartesian(K, newAlph, newBeta)
		newb = newLamb
		domain = GetDomain(K, newA, newb, 0.1, True)
		newuh, error_L2, error_max, = ModelforTest(domain)
		J = GetVal(ufl.dot(ufl.grad(uh), ufl.grad(uh)) * ufl.dx)
		newJ = GetVal(ufl.dot(ufl.grad(newuh), ufl.grad(newuh)) * ufl.dx)
		difJ = J-t*min([1./wA, 1./wB, 1./wL])*0.5*norm
		st = "{:08.6f}".format(t)
		sJ = "{:07.4f}".format(J)
		snewJ = "{:07.4f}".format(newJ)
		sdifJ = "{:07.4f}".format(difJ)
		print(f"t = {st}, J = {sJ}, NewJ = {snewJ}, Dif = {sdifJ}")
	
#Example02()	



def TestPolyhedral():
	pl = pentagonal_prism
	A = np.array(pl["A"])
	b = np.array(pl["b"])
	V = GetVertices(A, b, pl["vertices"])
	print("Vertices:")
	for v in V:
		print(v)
	print('Maximum cosine = {:4.2f}'.format(CheckVertices(V)))
	PlotPolyhedral(pl, 0.5)


def PlotResults():
	mtx = np.load("SaintVenant_indi2.npy")[1:, :]
	k = np.arange(mtx.shape[0])
	
	fig, axs = plt.subplots(2, 3, figsize=(15, 8))
	axs[0, 0].plot(k, mtx[:, 0], c = 'blue', lw = 3) # Lagrangian
	axs[0, 0].grid()
	
	axs[1, 0].plot(k, mtx[:, 7], c = 'blue', lw = 3) # J
	axs[1, 0].grid()
	
	axs[0, 1].plot(k, mtx[:, 2], c = 'red', lw = 3) # abs of lambda
	axs[0, 1].plot(k, mtx[:, 3], c = 'green', lw = 3) # abs of mu
	axs[0, 1].plot(k, mtx[:, 4], c = 'orange', lw = 3) # abs of z
	axs[0, 1].grid()
	
	axs[1, 1].plot(k, mtx[:, 1], c = 'blue', lw = 3) # norm of x
	axs[1, 1].grid()
	#axs[1].set_ylim([-0.2, 5.5])
	axs[0, 2].plot(k, mtx[:, 5], c = 'red')
	axs[0, 2].plot(k, mtx[:, 6], c = 'orange')
	axs[0, 2].set_yscale("log")
	axs[0, 2].grid()
	
	axs[1, 2].plot(k, mtx[:, 8], c = 'orange', lw = 3) # volume
	axs[1, 2].grid()
	
	fig.tight_layout()
	plt.show()
	
	mtx = np.load("SaintVenant_coor2.npy")
	K = int(mtx.shape[1]/3) 
	mtx = mtx[-1, :]
	Alph = mtx[0:K]
	Beta = mtx[K:2*K]
	Lamb = mtx[2*K:]
	X = SphericalToCartesian(K, Alph, Beta)
	s = GetDomain(12, X, Lamb, 0.2, True)
	#plt.savefig('img'+str(i)+'.png')
	

#Example04()
PlotResults()
#ReadFileDodecahedron()	
	
'''
	n = ufl.FacetNormal(domain)
	flux1 = ufl.dot(ufl.grad(uh), n)**2 * ufl.ds
	flux2 = ufl.dot(Guh, n)**2 * ufl.ds
	total_flux1 = fem.assemble_scalar(fem.form(flux1))
	total_flux2 = fem.assemble_scalar(fem.form(flux2))	
	print('Total :', total_flux1, total_flux2)

'''
