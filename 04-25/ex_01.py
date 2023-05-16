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
    tmp[i,:,:] = np.repeat(ro[np.newaxis,:,:],9,axis=0)[i,:,:] *weights[i]

f = tmp * (1 + np.moveaxis(3*np.dot(u,e.T),-1,0) + np.moveaxis((9/2)*np.dot(u,e.T)**2,-1,0) - 1.5*(u[:,:,0]**2 + u[:,:,1]**2))
#plt.imshow(u[:,:,0].T, origin='upper')
#plt.show()


# done

# here 
# will be loop
# inlet
f_eq = np.zeros((9,Nx,Ny))
for step in tqdm(range(20000)):
    # 1. Calculate the density and equilibrium distribution on the inlet
    #(always use initial velocity 𝐮𝟎 in this step).
    ro[0,:] = (2*(f[3,0,:]+f[6,0,:]+f[7,0,:]) + (f[0,0,:]+f[2,0,:]+f[4,0,:])) / (1 - (np.sqrt((u_0[0,:,0])**2 + u_0[0,:,1]**2)))
    #print(1 - np.abs(np.sqrt((u[0,:,0])**2 + u[0,:,1]**2)))
    #print(ro[0,80])


    
    for i in range(9):
        tmp[i,:,:] = np.repeat(ro[np.newaxis,:,:],9,axis=0)[i,:,:] *weights[i]

    f_eq[:,0,:] = tmp[:,0,:] * (1 + np.moveaxis(3*np.dot(u_0,e.T),-1,0)[:,0,:] + np.moveaxis(9/2*np.dot(u_0,e.T)**2,-1,0)[:,0,:] - 1.5*(u_0[:,:,0]**2 + u_0[:,:,1]**2)[0,:])

    #2. Apply the boundary conditions for the inlet and outlet.
    f[[1,5,8],0,:] = f_eq[[1,5,8],0,:]
    f[[3,6,7],Nx-1,:] = f[[3,6,7],Nx-2,:]

    # 3. Recalculate density and equilibrium distribution everywhere.

    ro = np.sum(f,axis=0)
    tmp_2 =  tmp[:,:,:] * (1 + np.moveaxis(3*np.dot(u,e.T),-1,0)[:,:,:] + np.moveaxis(9/2*np.dot(u,e.T)**2,-1,0)[:,:,:] - 1.5*(u[:,:,0]**2 + u[:,:,1]**2)[:,:])
    f_eq[:,1:,:] = tmp_2[:,1:,:]

    #4. collision step
    tmp_3 = np.copy(f)
    tmp_3 -= (f_eq / tau)
    f_col = f - tmp_3/tau 
    #5. Replace parts of the distribution function after collision that
    #correspond to fluid-solid boundaries with bounce-back condition
    #(you can just replace it everywhere inside the obstacle)

    #6. Calculate the streaming step 
    for i in range(9):
        new_f[i,:,:] = np.roll(np.roll(f_col[i,:,:],e[i][0],axis=0),e[i,1],axis=1)

    
    #7. Use distribution function after streaming as the initial one for the
    #next iteration (go back to 1).
    f = new_f
    print(f)
    tmp_1 = np.zeros(np.shape(u))
    for i in range(9):
        
        tmp_1[:,:,0] += e[i,0] * f[i]
        tmp_1[:,:,1] += e[i,1] * f[i]

    u = tmp_1
    u[:,:,0] /= ro
    u[:,:,1] /= ro
    #u = np.clip(u,0.001,1)

    if step % 100 == 0:
        plt.figure()
        plt.imshow((u[:,:,0]**2 + u[:,:,1]**2).T, vmin = 0.001, vmax=1)
        #plt.imshow(wedge.T)
        plt.title(f"u between {np.min(u[:,:,0]**2 + u[:,:,1]**2)} and {np.max(u[:,:,0]**2 + u[:,:,1]**2)}")
        plt.savefig(f"{step}.png")
        plt.close()