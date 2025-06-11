"""
V1: 2024/11/09
V2: 2024/11/16 periodic boundary condition and rectangular initial domain
V3: put the cahn-hilliard (transformed the equations into 2 coupled pdes) with growth
    also put the surface tension force in the momentum equation, 
    and active body force proportional to velocity direction of the tumor phase (phi=1). 
    Also added the estimated values of parameters.
    Generated interfacial instability.
V4: equations have been non-dimensionalized. Check the overleaf. (2024/11/28)
V5: The inertial term is zero. Box size=1, and all the dimensionless values are unity for the start. The P4 parameter has been comented out. In the weak form its replaced by 2
"""


from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import mshr



'''''''''''''''''
Params
'''''''''''''''''
# sim parameters
dt =0.005       # Smaller time step to improve convergence, old value: 0.005
num_steps = 200  # Increased number of steps due to reduced dt
t  = 0


#Mobility = 1.0
#Kappa_ch = 1.0
eta_s_t  = 1.0   # using the units Pa s. water has 0.001 Pa s
eta_s_m  = 1.0
#eta_p_t  = 1.0   # assuming solvent viscosity is the same as polymeric viscosity
#eta_p_m  = 1.0     
#lambda_t = 1.0   # 20241126
#lambda_m = 1.0   # 20241126
#rho_t    = 1.0
#rho_m    = 1.0
#alpha_t  = 1.0
#alpha_m  = 0.0
#division_rate = 0.1  # old value: 0.1 HY note: 1/division_rate is roughly the time scale?


#rho_mix=(rho_t+rho_m)/2
#eta_s_mix=(eta_s_t+eta_s_m)/2
#alpha_mix=(alpha_t+alpha_m)/2
#lambda_mix=(lambda_t+lambda_m)/2
#eta_p_mix=(eta_p_t+eta_p_m)/2


kappa_p = 1e3 # bulk modulus (assign a large number)
epsilon  = 0.01  # old: 0.01. Interface thickness parameter
P1=1.0
P2=1.0
P3=0.1
#P4=2.0
P5=0.01
P6=1.0    #p6 need not be close 1. explore 1 , 10, 100 
growth_rate_parameter=10.0 #earlier it was 10
surface_tension_parameter=9.0 #earlier it was 1e-6





# material_parameters   #these are starred material parameters wrt to latex file
def eta_s(phi):
    return 1.0 # Tumor has higher eta_s, phi=1 (subscript 't' implies tumor and 'm' implies matrix)

def eta_p(phi):
    return  eta_p_m # Non-tumor has higher eta_p, phi=-1

def lambda_(phi):
    return lambda_m # Non-Tumor has higher lambda_, phi=1

def rho(phi):
    return 1 # Non-Tumor has higher rho, phi=1

def alpha(phi):
    return (1+phi)/2.0  # Tumor has higher alpha, phi=1

def growth_rate(phi):
    return growth_rate_parameter*exp(-phi**2.0)*(1-phi**2.0)

'''''''''''''''''
Create mesh and mark boundaries
'''''''''''''''''
# Option 1
# mesh = UnitSquareMesh(16, 16)
# Option 2
box_length=1.0
domain = mshr.Rectangle(Point(0.0, 0.0), Point(box_length, box_length))
mesh   = mshr.generate_mesh(domain, 48)


x = SpatialCoordinate(mesh)

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],0.0) and on_boundary
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],box_length) and on_boundary
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],0.0) and on_boundary
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],box_length) and on_boundary

# Mark boundary subdomains
facets = MeshFunction("size_t", mesh, 1)
DomainBoundary().mark(facets, 1)
Left().mark(facets, 2)
Right().mark(facets,3)
Bottom().mark(facets, 4)
Top().mark(facets,5)

#
#plot(mesh)

'''''''''''''''''
FEM setup
'''''''''''''''''
# Left-Right: Periodic boundary condition
#class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
#    def inside(self, x, on_boundary):
#        return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)

    # Map right boundary (H) to left boundary (G)
#    def map(self, x, y):
#        y[0] = x[0] - 1.0
#        y[1] = x[1]

        

# function space
U = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
T = TensorElement("Lagrange", mesh.ufl_cell(), 1)

ELEMENT = MixedElement([U, P, P, P, T]) # velocity u,  pressure p, phase phi, chem_pot, polymer stress tensor tau
ME = FunctionSpace(mesh, ELEMENT)

W1 = FunctionSpace(mesh, P) # scalar space for visualization
W2 = FunctionSpace(mesh, U) # vector space for visualization
W3 = FunctionSpace(mesh, T) # tensor space for visualization

# test functions
w_test = TestFunction(ME)
(u_test, p_test, phi_test, chem_test, tau_test) = split(w_test)

# actual functions
w = Function(ME)
(u, p, phi, chem_pot, tau) = split(w)

# trial function for automatic derivative
dw = TrialFunction(ME)

# for storing previous steps
# w_n = Function(ME)
# (u_n, phi_n, p_n, tau_n) = split(w_test)


'''''''''''''''''
Initial conditions
'''''''''''''''''
# for storing previous steps
u_state   = FunctionSpace(mesh, U)
u_n       = Function(u_state)
phi_state = FunctionSpace(mesh, P)
phi_n     = Function(phi_state)
tau_state = FunctionSpace(mesh, T)
tau_n     = Function(tau_state)

# p_n.assign(Constant(0.0))         # HY note: This doesn't seem necessary unless we assign an evolution equation for p
# u_n.assign(Constant((0.1, 0.1)))  # Small initial velocity
# tau_n.assign(Constant(((0.01, 0.01), (0.01, 0.01))))  # Small initial stress

# HY note: need to think more about initial condition
u_n.assign(Constant((0.01, 0.01)))
tau_n.assign(Constant(((0.0, 0.0), (0.0, 0.0))))

class InitialCondition(UserExpression):
    def __init__(self, R=0.20, epsilon_=0.005, center=(box_length/2, box_length/2), **kwargs):
        super().__init__(**kwargs)
        self.R = R
        self.epsilon_ = epsilon_
        self.center = center

    def eval(self, values, x):
        # Compute radius from the specified center
        r = np.sqrt((x[0] - self.center[0])**2 + (x[1] - self.center[1])**2)
        
        # Smooth tanh transition at r = R
        # sqrt(2) is sometimes included in phase-field formulations to match
        # certain theoretical profiles, but it's not strictly required. 
        # If you want the classical form, use:  ( self.R - r ) / ( np.sqrt(2)*self.epsilon_ )
        values[0] = np.tanh((self.R - r) / (self.epsilon_))

    def value_shape(self):
        return ()

radius = 0.20
epsilon_ = 0.005
center = (box_length/2, box_length/2)
phi_init = InitialCondition(degree=2, R=radius, epsilon_=epsilon_, center=center)
phi_n.assign(project(phi_init, phi_state))
# phi.interpolate(InitialCondition(degree=1))
# phi_n.assign(phi)
#plot(phi_n)
#plt.figure(figsize=(6,6))
#ax = plot(phi_n,cmap='viridis')
#plt.colorbar(ax,label='$\phi$')
#plt.show()


'''''''''''''''''
Utils
'''''''''''''''''

def local_project(v, V, u=None):
    if V.ufl_element().degree() ==1:
        dv = TrialFunction(V)
        v_ = TestFunction(V)
        a_proj = inner(dv, v_)*dx
        b_proj = inner(v, v_)*dx
        Lsolver = LocalSolver(a_proj, b_proj)
        Lsolver.factorize
        if u is None:
            u = Function(V)
            Lsolver.solve_local_rhs(u)
            return u
        else:
            Lsolver.solve_local_rhs(u)
            return
    else:
        u = project(v,V)
        return u


'''''''''''''''''
Weak forms
'''''''''''''''''
# Body force (proportional to velocity direction)
#unit_velocity= u/(sqrt(dot(u, u)) + 1e-8)  #divide by a small number to prevent division by zero
# f_body = alpha(phi)*unit_velocity  # Corrected the sign of the body force
f_body = P2*alpha(phi_n)*u_n # 20241126: let's keep it simple for now

# Define the stress tensor
D = sym(grad(u))
# sigma = -p * Identity(len(u)) + 2 * eta_s(phi) * D + tau
sigma = -p * Identity(len(u)) + P1*eta_s(phi_n)*D + (0.01)*(1-phi_n)*tau_n/2 # 20241126 +p  #multiplied by P2 the deviatoric part, i think it should be -p

# Variational form for momentum balance (mixed formulation)
R_u = ( inner(sigma, grad(u_test)) * dx  # Stress tensor applied to gradient of test function
       -dot(f_body, u_test) * dx  # Body force term
       +surface_tension_parameter*dot(chem_pot*grad(phi),u_test)*dx
       ) #the surface tension force 20241126: I think we need a pre-factor coefficient here


#variational form for the polymeric stress# nabla_grad(tau)?    #used P3 and P4 in tau variational forms
#R_tau = ((P3/dt)*inner((tau - tau_n), tau_test) * dx
#         + P3*inner(dot(u, nabla_grad(tau)), tau_test) * dx     #should this be grad and nabla_grad? 
#         - P3*inner(dot(tau, grad(u)), tau_test) * dx
#         - P3*inner(dot(grad(u).T, tau), tau_test) * dx
#         - (2.0) * inner(D, tau_test)* dx
#         + inner(tau, tau_test) * dx
#        )

R_tau = ((P3/dt)*inner((tau - tau_n), tau_test) * dx
         - (2.0) * inner(D, tau_test)* dx
         + 1.0*inner(tau, tau_test) * dx
        )

# Variational form for the cahn-hilliard equation   #used P5 in this variational form
R_phi = (inner((phi - phi_n), phi_test) * dx
         + 10.0*dt*inner(dot(u_n, grad(phi)), phi_test) * dx
         + dt*P5*dot(grad(chem_pot), grad(phi_test)) * dx
         - dt*growth_rate(phi) * phi_test * dx
         )

R_chem_pot=(chem_pot*chem_test*dx-P6*(phi**3.0 - phi)*chem_test*dx #used P6 in this variational form
            -P6*(epsilon**2)*dot(grad(phi), grad(chem_test)) * dx
           )


# incompressibility condition (a slightly compressible formulation)
# HY note: if we have pressure as an indepdendent field, we need to give its own weak form
R_p  = (tr(D)+p/kappa_p)*p_test*dx   #made it +p/kappa

# Total redidue
# HY note: let's solve it as a global weak form instead of one by one
R_tot =  R_u +R_tau + R_phi + R_chem_pot+ R_p

# tangent
a = derivative(R_tot, w, dw)



'''''''''''''''''
Boundary conditions
'''''''''''''''''
# HY note: think a bit more about bcs here
bcs_1 = DirichletBC(ME.sub(0), Constant((0.0,0.0)), facets, 4) # bottom
bcs_2 = DirichletBC(ME.sub(0), Constant((0.0,0.0)), facets, 5) # top
bcs_3 = DirichletBC(ME.sub(0), Constant((0.0,0.0)), facets, 2) # bottom
bcs_4 = DirichletBC(ME.sub(0), Constant((0.0,0.0)), facets, 3) # top
bcs = [bcs_1, bcs_2, bcs_3, bcs_4]
#bcs=[]




'''''''''''''''''
Run
'''''''''''''''''

# solver
# stressProblem = NonlinearVariationalProblem(R_tot, w, [], J=a) # HY note: for now, we don't have boundary condition.
stressProblem = NonlinearVariationalProblem(R_tot, w, bcs, J=a) # 2024/11/16
solver = NonlinearVariationalSolver(stressProblem)
prm = solver.parameters
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = 'mumps' # 'mumps' # 'petsc'   #'gmres'
prm['newton_solver']['absolute_tolerance'] =  1.e-8
prm['newton_solver']['relative_tolerance'] =  1.e-8
prm['newton_solver']['maximum_iterations'] = 100

# Prepare for movie
fig, ax = plt.subplots(figsize=(6, 6))
phi_plot = plot(phi_n, cmap='viridis', vmin=-1, vmax=1)
plt.colorbar(phi_plot, label='$\phi$')
metadata = dict(title="Simulation Movie", artist="FEniCS")
writer = PillowWriter(fps=15)





print("------------------------------------")
print("Simulation Start")
print("------------------------------------")

idx = 0 # counter
#with writer.saving(fig, "phi_simulation.gif", dpi=100):
for n in range(num_steps):
    t += dt
    

    (iter, converged) = solver.solve()
    
    # update old fields
    u_n.assign(local_project(u,u_state))
    phi_n.assign(local_project(phi,phi_state))
    tau_n.assign(local_project(tau,tau_state))
    
    # output every a few steps
    if idx%20==0:
        print(f'Step:{idx}/{num_steps}')
        plt.figure(figsize=(12,6))
        plt.subplot(121)
        ax = plot(phi_n,cmap='viridis')
        plt.colorbar(ax,label='$\phi$')
        
        plt.subplot(122)
        ax = plot(u,cmap='viridis',scale=0.1)
        plt.colorbar(ax,label='$u$')
        plt.show()

    idx += 1

        #if  idx%5 == 0:  # Save every 20th frame
        #    print(f"Step: {n}/{num_steps}")
        #    phi_plot.set_array(phi_n.compute_vertex_values(mesh))
        #    writer.grab_frame()
#print("Simulation complete. Movie saved as phi_simulation.mp4.")


        