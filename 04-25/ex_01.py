import numpy as np
from matplotlib import pyplot as plt
# constances

Nx = 520
Ny = 180

e = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
)

ro = np.ones((Nx, Ny))
u_0 = np.zeros((Nx, Ny, 2))
u = np.zeros((Nx, Ny, 2))
f = np.zeros((9, Nx, Ny))
u = np.zeros((Nx, Ny, 2))
u_in = 0.04
epsilon = 0.0001

weights = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36])

def calc_density(f):
    new_ro = np.sum(f, axis=0)
    return new_ro


def calc_velocity(ro, f, e):
    u = np.zeros((Nx, Ny, 2))
    u[:, :, 0] = np.sum(f * e[:, 0], axis=0)
    u[:, :, 1] = np.sum(f * e[:, 1], axis=0)
    return u / ro


# initialization

u_0[:, :, 0] = u_in + u_in * epsilon * np.sin(np.arange(Ny) * 2 * np.pi / (Ny - 1))
u = np.copy(u_0)
tmp = np.zeros((9,Nx,Ny))
#for i in range(len(e)):
for i in range(9):
    tmp[i,:,:] = np.repeat(ro[np.newaxis,:,:],9,axis=0)[i,:,:] *weights[i]

f = tmp * (1 + np.moveaxis(3*np.dot(u,e.T),-1,0) + np.moveaxis(9/2*np.dot(u,e.T)**2,-1,0) - 1.5*(u[:,:,0]**2 + u[:,:,1]**2))
#plt.imshow(u[:,:,0].T, origin='upper')
#plt.show()


# done

# here will be loop
# inlet
f_eq = np.zeros((9,Nx,Ny))

# 1. Calculate the density and equilibrium distribution on the inlet
#(always use initial velocity 𝐮𝟎 in this step).
ro[0,:] = (2*(f[3,0,:]+f[6,0,:]+f[7,0,:]) + (f[0,0,:]+f[2,0,:]+f[4,0,:])) / (1 - (np.sqrt(u[0,:,0]**2 + u[0,:,1]**2)))
print(ro.shape)

for i in range(9):
    tmp[i,:,:] = np.repeat(ro[np.newaxis,:,:],9,axis=0)[i,:,:] *weights[i]

f_eq[:,0,:] = tmp[:,0,:] * (1 + np.moveaxis(3*np.dot(u_0,e.T),-1,0)[:,0,:] + np.moveaxis(9/2*np.dot(u_0,e.T)**2,-1,0)[:,0,:] - 1.5*(u_0[:,:,0]**2 + u_0[:,:,1]**2)[0,:])

#2. Apply the boundary conditions for the inlet and outlet.
f[[1,5,8],0,:] = f_eq[[1,5,8],0,:]
f[[3,6,7],:,:] = np.roll(f_eq,-1,axis=1)[[1,5,8],:,:] #to jest mega.

# 3. Recalculate density and equilibrium distribution everywhere.

ro = np.sum(f,axis=0)
f_eq[:,:,:] = tmp[:,:,:] * (1 + np.moveaxis(3*np.dot(u_0,e.T),-1,0)[:,:,:] + np.moveaxis(9/2*np.dot(u_0,e.T)**2,-1,0)[:,:,:] - 1.5*(u_0[:,:,0]**2 + u_0[:,:,1]**2)[:,:])


#5. Replace parts of the distribution function after collision that
#correspond to fluid-solid boundaries with bounce-back condition
#(you can just replace it everywhere inside the obstacle)




#6. Calculate the streaming step



#7. Use distribution function after streaming as the initial one for the
#next iteration (go back to 1).
print("aa")