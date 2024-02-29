import gmsh

import numpy as np

from mpi4py import MPI

from dolfinx.io import gmshio
from dolfinx.plot import vtk_mesh
from dolfinx.fem.petsc import LinearProblem
from dolfinx import fem, mesh, default_scalar_type

import ufl

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
	
def CreatePolyhedral(A, b, K, size = .5):
	for k in range(1, K+1):
		CreateClosedHalfPlane(gmsh, A[k-1], b[k-1], k)
	IntersectionOf(gmsh, K)
	AddNames(gmsh, K)
	SetSizeMesh(gmsh, size)
	gmsh.model.mesh.generate() #create the mesh

def normalize(A, b):
	normA = np.linalg.norm(A, ord = 2, axis = 1)
	A = A/normA[:,None]
	b = b/normA
	return A, b

def GetDomain(A, b, K):
	gmsh.initialize()
	CreatePolyhedral(A, b, K)
	#gmsh.fltk.run()
	domain, cell_tags, facet_tags = \
		gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)	
	gmsh.clear()
	gmsh.finalize()
	return domain


def SolvePDE01(domain):
	# solve PDE
	V = fem.FunctionSpace(domain, ("Lagrange", 1))
	f = fem.Constant(domain, default_scalar_type(10.))
	# or
	'''
	x = ufl.SpatialCoordinate(domain)
	C0 = fem.Constant(domain, default_scalar_type(0.5))
	C1 = fem.Constant(domain, default_scalar_type(0.2))
	C2 = fem.Constant(domain, default_scalar_type(0.3))
	beta = fem.Constant(domain, default_scalar_type(10))
	f = -10. * ufl.exp(-beta*((x[0]-C0)**2 + (x[1]-C1)**2 + (x[2]-C2)**2))
	'''
	uD = fem.Function(V)
	uD.interpolate(lambda x: x[2]*x[1]*x[0]*0)
	domain.topology.create_connectivity(d0 = 2, d1 = 3)
	boundary_facets = mesh.exterior_facet_indices(domain.topology)
	boundary_dofs = fem.locate_dofs_topological(V, 2, boundary_facets)
	bc = fem.dirichletbc(uD, boundary_dofs)
	
	u = ufl.TrialFunction(V)
	v = ufl.TestFunction(V)

	a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
	L = f * v * ufl.dx
	problem = LinearProblem(a, L, bcs = [bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
	uh = problem.solve()

	W = fem.VectorFunctionSpace(domain, ("Lagrange", 1))
	w = ufl.TrialFunction(W)
	v = ufl.TestFunction(W)

	a = ufl.inner(w, v) * ufl.dx
	L = ufl.inner(ufl.grad(uh), v) * ufl.dx
	problem = LinearProblem(a, L)
	Guh = problem.solve()

	n = ufl.FacetNormal(domain)
	flux1 = ufl.dot(ufl.grad(uh), n)**2 * ufl.ds
	flux2 = ufl.dot(Guh, n)**2 * ufl.ds
	total_flux1 = fem.assemble_scalar(fem.form(flux1))
	total_flux2 = fem.assemble_scalar(fem.form(flux2))	
	print('Total :', total_flux1, total_flux2)
	return V, uh, W, Guh

def PlotSolution(V, uh, W, Guh, domain):
	
	plotter = pyvista.Plotter()
	grid = pyvista.UnstructuredGrid(*vtk_mesh(V))
	grid.point_data["u"] = uh.x.array
	grid.set_active_scalars("u")
	warp = grid.warp_by_scalar("u", factor=1)
	actor = plotter.add_mesh(warp, show_edges=True)
	plotter.show()
	
	plotter = pyvista.Plotter()
	plotter.set_position([0, 0, 5])

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
	actor = plotter.add_mesh(grid, style="wireframe", color="k")
	actor2 = plotter.add_mesh(glyphs)

	plotter.show()

def Energy(u):
	# return the energy of the approximate solution
	energy = .5*ufl.dot(ufl.grad(u), ufl.grad(u))*ufl.dx
	return fem.assemble_scalar(fem.form(energy))

def GetGrad(domain, uh):
	W = fem.VectorFunctionSpace(domain, ("DG", 0))
	Guh = fem.Function(W)
	G_exp = fem.Expression(ufl.grad(uh), W.element.interpolation_points())
	Guh.interpolate(G_exp)
	return Guh

def example():
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
	A, b = normalize(A, b)
	
	domain = GetDomain(A, b, 8)
	V, uh, W, Guh = SolvePDE01(domain)

	print('Energy = ', Energy(uh))
	
	p = pyvista.Plotter()
	topology, cell_types, geometry = vtk_mesh(V)
	grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
	grid["Gu"] = Guh.x.array.reshape((geometry.shape[0], 3))
	actor_0 = p.add_mesh(grid, style="wireframe", color="k")
	warped = grid.warp_by_vector("Gu", factor=0.01)
	actor_1 = p.add_mesh(warped, show_edges=True)
	p.show_axes()
	p.show()
	#https://jsdokken.com/dolfinx-tutorial/chapter3/robin_neumann_dirichlet.html
	#PlotSolution(V, uh, W, Guh, domain)
	
	
example()	

	
