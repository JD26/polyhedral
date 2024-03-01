import numpy as np

import gmsh

import ufl

from dolfinx.io import gmshio, XDMFFile
from dolfinx.plot import vtk_mesh
from dolfinx.fem.petsc import LinearProblem
from dolfinx import fem, mesh, default_scalar_type

from mpi4py import MPI

import pyvista


def CreateClosedHalfPlane(gmsh, n, a, k, c = 100):
	# create a closed half-plane H = {x | <n,x> <= a}
	# 'n' is a u unit vector
	# 'a' must be positive ==> 0 in H
	# tag k must be greater than or equal to 1
	p, q = a*n, -c*n
	gmsh.model.occ.addCylinder(p[0], p[1], p[2], q[0], q[1], q[2],  c, k)
	
def IntersectionOf(gmsh, K):
	# intersect the closed half-planes
	# K is the number of closed half-spaces
	# the tags are K+1, ..., 2K-1 
	gmsh.model.occ.intersect([(3, 1)], [(3, 2)], K+1)
	for i in range(1, K-1):
		gmsh.model.occ.intersect([(3, K+i)], [(3, 2+i)], K+i+1)
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
	A = A/normA[:,None]
	b = b/normA
	return A, b

def GetDomain(A, b, K, size): 
	gmsh.initialize()
	CreatePolyhedral(gmsh, A, b, K, size)
	#gmsh.fltk.run() # Plot the mesh
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
	# Each fk function must accept the set of all nodes x
	# and return true or false for each element of x 
	x = ufl.SpatialCoordinate(domain)
	faces = []
	for k in range(K):
		fk = lambda x : np.isclose(np.inner(x.T, A[k])-b[k], 0)
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
	domain.topology.create_connectivity(domain.topology.dim-1, domain.topology.dim)
	with XDMFFile(domain.comm, "facet_tags.xdmf", "w") as xdmf:
		xdmf.write_mesh(domain)
		xdmf.write_meshtags(facet_tag, domain.geometry)

	return facet_tag

def dsXFace(domain, A, b, K):
	faces = TagAndFaces(domain, A, b, K)
	facet_tag = MarkedFaces(domain, faces)
	return ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)

def DuXn2(uh, ds, K):
	d = np.zeros(K)
	for k in range(K):
		int_k = ufl.dot(ufl.grad(uh), ufl.grad(uh)) * ds(k+1)
		d[k] = fem.assemble_scalar(fem.form(int_k))
	return d

def DuXn2_xXv(uh, ds, K, domain, v):
	d = np.zeros(K)
	x = ufl.SpatialCoordinate(domain)
	for k in range(K):
		v0 = fem.Constant(domain, default_scalar_type(v[k][0]))
		v1 = fem.Constant(domain, default_scalar_type(v[k][1]))
		v2 = fem.Constant(domain, default_scalar_type(v[k][2]))
		xXv = x[0]*v0 + x[1]*v1 + x[2]*v2
		int_k = xXv*ufl.dot(ufl.grad(uh), ufl.grad(uh)) * ds(k+1)
		d[k] = fem.assemble_scalar(fem.form(int_k))
	return d

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

def example():
	
	size_mesh = 0.1
	K, A, b = Polyedral01()
	domain = GetDomain(A, b, K, size_mesh)

	V, uh = SolvePDE01(domain)

	print('Energy = ', Energy(uh))

	#PlotSolution(domain, V, uh)
	#PlotGrad0(domain, V, uh)
	#PlotGrad1(domain, V, uh)
	
	ds = dsXFace(domain, A, b, K)
	v = np.random.rand(K, 3)
	d = DuXn2(uh, ds, K)
	d = DuXn2_xXv(uh, ds, K, domain, v)
	print(d)

example()	

	
'''
	n = ufl.FacetNormal(domain)
	flux1 = ufl.dot(ufl.grad(uh), n)**2 * ufl.ds
	flux2 = ufl.dot(Guh, n)**2 * ufl.ds
	total_flux1 = fem.assemble_scalar(fem.form(flux1))
	total_flux2 = fem.assemble_scalar(fem.form(flux2))	
	print('Total :', total_flux1, total_flux2)

'''