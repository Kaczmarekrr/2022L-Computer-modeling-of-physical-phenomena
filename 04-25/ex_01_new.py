#%%
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
# constances

Nx = 520
Ny = 180



e = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
)

e_reverse = np.array(
    [[0, 0], [-1, 0], [0, -1], [1, 0], [0, 1],[-1, -1], [1, -1], [1, 1], [-1, 1]]
)
ro = np.ones((Nx, Ny),dtype=np.float64)
u_0 = np.zeros((Nx, Ny, 2),dtype=np.float64)
u = np.zeros((Nx, Ny, 2),dtype=np.float64)
f = np.zeros((9, Nx, Ny),dtype=np.float64)
new_f = np.copy(f)
u = np.zeros((Nx, Ny, 2),dtype=np.float64)
u_in = 0.04
epsilon = 0.0001

Re = 220
v_lb = (u_in* (180/2)) / Re
tau = 3*v_lb + 1/2
print(tau)


weights = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36])
wedge = np.fromfunction(lambda x, y: np.abs(x-Nx/4) + np.abs(y) < Ny/2, (Nx,Ny)) #good wedge
# initialization
wedge[:,0] = True
wedge[:,-1] = True






u_0[:, :, 0] = u_in + u_in * epsilon * np.sin(np.arange(Ny) * 2 * np.pi / (Ny - 1))
u = np.copy(u_0)
tmp = np.zeros((9,Nx,Ny))
#for i in range(len(e)):
for i in range(9):
    f[i,:,:] = weights[i]*ro * (1 + np.dot(u[:,:],e[i]) + 4.5*np.dot(u[:,:],e[i])**2 - 1.5*(u[:,:,0]**2 + u[:,:,1]**2))
plt.imshow(u[:,:,0].T, origin='upper')
plt.show()

print(f)

# done

# here 
# will be loop
# inlet
f_eq = np.zeros((9,Nx,Ny))
# one step in loop
for step in tqdm(range(1)):
    ro[0,:] = (2*(np.sum(f[[3,6,7],0,:],axis=0)) + np.sum(f[[0,2,4],0,:],axis=0))/ (1 - (np.sqrt((u_0[0,:,0])**2 + u_0[0,:,1]**2)))




    for i in range(9):
        f_eq[i,0,:] = weights[i]*ro[0,:] * (1 + np.dot(u_0[0,:],e[i]) + 4.5*np.dot(u_0[0,:],e[i])**2 - 1.5*(u_0[0,:,0]**2 + u_0[0,:,1]**2))

    #2. Apply the boundary conditions for the inlet and outlet.
    f[[1,5,8],0,:] = f_eq[[1,5,8],0,:]
    f[[3,6,7],Nx-1,:] = f[[3,6,7],Nx-2,:]


    ro = np.sum(f,axis=0)

    """    plt.imshow(ro.T)
    plt.colorbar()
    plt.savefig(f"ro_{step}.png")"""

    tmp_1 = np.zeros(np.shape(u))
    for i in range(9):
        
        tmp_1[:,:,0] += e[i,0] * f[i,:,:] / ro
        tmp_1[:,:,1] += e[i,1] * f[i,:,:] / ro

    u = tmp_1
#%%
    for i in range(9):
        f_eq[i,:,:] = weights[i]*ro[:,:] * (1 + np.dot(u[:,:],e[i]) + 4.5*np.dot(u[:,:],e[i])**2 - 1.5*(u[:,:,0]**2 + u[:,:,1]**2))

    #4. collision step
    """tmp_3 = np.copy(f)
    tmp_3 -= (f_eq / tau)
    f_col = f - tmp_3/tau 
    """
    f_col = f.copy()

    for i in range(9):
        tmp_22 = (f[i,:,:]-f_eq[i,:,:])/tau
        f_col[i] -= tmp_22
    # collitions
    #for i in range(9):
    #    f_col[i,wedge] = -f_col[i,wedge]


    # streaming
    for i in range(9):
        new_f[i,:,:] = np.roll(np.roll(f_col[i,:,:],e[i][0],axis=0),e[i,1],axis=1)

    f = new_f

    #u = np.where(u < 0.00001,0.00001,u)

    #u[wedge,:] = 0
    if step % 100 == 0:
        plt.figure()
        plt.imshow((u[:,:,0]**2 + u[:,:,1]**2).T)#, vmin = 0.001, vmax=1)
        #plt.imshow(wedge.T)
        plt.title(f"u between {np.min(u[:,:,0]**2 + u[:,:,1]**2)} and {np.max(u[:,:,0]**2 + u[:,:,1]**2)}")
        plt.savefig(f"{step}.png")
        plt.close()