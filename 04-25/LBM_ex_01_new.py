#%%
import argparse
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

# constances

def main():
    args = getArgs()

    Nx = 520
    Ny = 180
    max_loops = 20000

    for_print = np.array([0,0])

    e = np.array(
        [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
    )

    e_reverse = np.array(
        [[0, 0], [-1, 0], [0, -1], [1, 0], [0, 1],[-1, -1], [1, -1], [1, 1], [-1, 1]]
    )

    reverse_ids = [0, 3, 4, 1, 2, 7, 8, 5, 6]
    ro = np.ones((Nx, Ny),dtype=np.float64)
    u_0 = np.zeros((Nx, Ny, 2),dtype=np.float64)
    u = np.zeros((Nx, Ny, 2),dtype=np.float64)
    f = np.zeros((9, Nx, Ny),dtype=np.float64)
    new_f = np.copy(f)
    u = np.zeros((Nx, Ny, 2),dtype=np.float64)
    u_in = 0.04
    epsilon = 0.0001

    Re = args.Re
    v_lb = (u_in* (180/2)) / Re
    tau = 3*v_lb + 1/2
    print(tau)


    weights = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36])
    wedge = np.fromfunction(lambda x, y: np.abs(x-Nx/4) + np.abs(y) < Ny/2, (Nx,Ny)) #good wedge
    wedge2 = np.fromfunction(lambda x, y: np.abs(x-Nx/4) + np.abs(y) < (Ny-2)/2, (Nx,Ny)) #good wedge
    # initialization
    #wedge[:,:] = False
    wedge = np.where(wedge2 == True,False,wedge)
    #print(np.where(wedge==True)[0])

    #wedge[:,:] = False
    #wedge[100,0:80] = True
    #wedge[100,100:Ny] = True
    #wedge[:,0] = True
    #wedge[:,-1] = True




    u_0[:, :, 0] = u_in + u_in * epsilon * np.sin(np.arange(Ny) * 2 * np.pi / (Ny - 1))
    u_0[wedge2,:] = 0

    u = np.copy(u_0)
    tmp = np.zeros((9,Nx,Ny))
    #for i in range(len(e)):
    for i in range(9):
        f[i,:,:] = weights[i]*ro * (1 + 3*np.dot(u[:,:],e[i]) + 4.5*np.dot(u[:,:],e[i])**2 - 1.5*(u[:,:,0]**2 + u[:,:,1]**2))



    # done

    # here 
    # will be loop
    # inlet
    f_eq = np.zeros((9,Nx,Ny))
    # one step in loop
    for step in tqdm(range(max_loops)):
        ro[0,:] = (2*(np.sum(f[[3,6,7],0,:],axis=0)) + np.sum(f[[0,2,4],0,:],axis=0))/ (1 - (np.sqrt((u_0[0,:,0])**2 + u_0[0,:,1]**2)))




        for i in range(9):
            f_eq[i,0,:] = weights[i]*ro[0,:] * (1 + 3*np.dot(u_0[0,:],e[i]) + 4.5*np.dot(u_0[0,:],e[i])**2 - 1.5*(u_0[0,:,0]**2 + u_0[0,:,1]**2))

        #2. Apply the boundary conditions for the inlet and outlet.
        f[[1,5,8],0,:] = f_eq[[1,5,8],0,:]
        f[[3,6,7],Nx-1,:] = f[[3,6,7],Nx-2,:]


        ro = np.sum(f,axis=0)
        #if np.any(ro < 0):
        #    print("wprong ro")
        #    for_print = np.where(ro<0)
        #    #break
            
        #print(step,ro[0,15],"ro")

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
            f_eq[i,:,:] = weights[i]*ro[:,:] * (1 + 3*np.dot(u[:,:],e[i]) + 4.5*np.dot(u[:,:],e[i])**2 - 1.5*(u[:,:,0]**2 + u[:,:,1]**2))

        #4. collision step
        """tmp_3 = np.copy(f)
        tmp_3 -= (f_eq / tau)
        f_col = f - tmp_3/tau 
        """
        f_col = np.copy(f)

        for i in range(9):
            tmp_22 = (f[i,:,:]-f_eq[i,:,:])/tau
            f_col[i] -= tmp_22
        #collitions
        

        #for i in range(1,9):
        #    f_col[i,wedge] = np.dot(f_col[i,wedge],e[i])
        for i in range(9):
            f_col[i,wedge] = f_col[reverse_ids[i],wedge]

        # streaming
        for i in range(9):
            new_f[i,:,:] = np.roll(np.roll(f_col[i,:,:],e[i][0],axis=0),-e[i,1],axis=1)

        new_f[:,:,0] = f_col[:,:,-1]
        new_f[:,:,-1] = f_col[:,:,0]
        #u = np.where(u < 0.00001,0.00001,u)

        #u[wedge,:] = 0
        if step % 100 == 0:
            plt.figure()
            plt.imshow((u[:,:,0]**2 + u[:,:,1]**2).T)#, vmin = 0.001, vmax=1)
            #plt.imshow(wedge.T)
            #plt.plot(f_eq[:,15,0])
            #plt.plot(f_col[:,15,0])
            #plt.plot(new_f[:,15,0])
            #plt.imshow(ro.T)

            #plt.title(f"u between {np.min(u[:,:,0]**2 + u[:,:,1]**2)} and {np.max(u[:,:,0]**2 + u[:,:,1]**2)}")
            plt.title(f"u between {np.min(ro)} and {np.max(ro)}")
            plt.plot(np.where(wedge2==True)[0], np.where(wedge2==True)[1], marker=".", markersize=1,color="k")
            #plt.plot(for_print[0],for_print[1],"r.")
            plt.savefig(f"{args.out_dir}/{step}.png")
            plt.close()

        f = new_f



def getArgs(argv=None):
    parser = argparse.ArgumentParser(description="vortex simulator")
    parser.add_argument("--out_dir", type=str, help="output_directory (which is created)")
    parser.add_argument("--Re", type=int, help="Re value")
    return parser.parse_args(argv)

if __name__ == "__main__":
    main()