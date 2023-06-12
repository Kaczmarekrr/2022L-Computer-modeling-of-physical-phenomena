import numpy as np
from matplotlib import pyplot as plt
import random
from tqdm import tqdm

# tracking them as list of individuals 
class individuals:
    def __init__(self,x,y,rep_timeout=3):
        self.x = x
        self.y = y
        self.rep_timeout = rep_timeout
        self.alife = 1
    
    

class predator(individuals):
    def __init__(self, x, y,rep_timeout=20,HP=3):
        super().__init__(x, y, rep_timeout=20)
        self.rep_timeout = rep_timeout
        self.HP = HP
    def __str__(self):
        return f"shark {self.x},{self.y}"
    
    def move(self,occupation_matrix):
        grid_size = 40
        e = np.array(
        [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
    )
        possible_move = np.zeros(8,dtype=int)
        for i in range(8):

            possible_move[i] = occupation_matrix[self.x,self.y] - np.roll(np.roll(occupation_matrix,-e[i][0],axis=0),-e[i,1],axis=1)[self.x,self.y]

        #print(occupation_matrix)
        if np.any(possible_move==1):
            move = np.random.choice(np.where(possible_move==1)[0])
            status = 2
            self.HP = 3
            self.rep_timeout-=1
            if self.rep_timeout <= 0:
                self.rep_timeout = 20
                status = 3
            
        elif np.any(possible_move==2):
            move = np.random.choice(np.where(possible_move==2)[0])
            status = 0
            self.HP -=1
            self.rep_timeout -= 1
            if self.rep_timeout <= 0:
                self.rep_timeout = 20
                status = 1
        else:
            status = 0
            self.HP -=1
            self.rep_timeout -= 1
            return status,occupation_matrix

        occupation_matrix[self.x,self.y] = 0
        
        #boundary condition
        self.x += e[move,0]
        self.y += e[move,1]
        if self.x < 0:
            self.x = grid_size-1
        elif self.x >= grid_size :
            self.x = 0

        if self.y < 0:
            self.y = grid_size-1
        elif self.y >= grid_size :
            self.y = 0

        occupation_matrix[self.x,self.y] = 2
    
        return status,occupation_matrix
class prey(individuals):
    def __init__(self, x, y,rep_timeout=3):
        super().__init__(x, y,rep_timeout=3)
        self.rep_timeout = rep_timeout
    def __str__(self):
        return f"fish {self.x},{self.y}"
    
    def move(self,occupation_matrix):
        grid_size = 40
        e = np.array(
        [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
    )
        possible_move = np.zeros(8,dtype=int)
        for i in range(8):

            possible_move[i] = occupation_matrix[self.x,self.y] - np.roll(np.roll(occupation_matrix,-e[i][0],axis=0),-e[i,1],axis=1)[self.x,self.y]
        

        #print(self.x,self.y,possible_move)
        if np.any(possible_move>0):
            random_move = np.random.choice(np.where(possible_move>0)[0])
            occupation_matrix[self.x,self.y] = 0
            #print(random_move)
            # x,y update 
            
            #boundary condition
            self.x += e[random_move,0]
            self.y += e[random_move,1]
            if self.x < 0:
                self.x = grid_size-1
            elif self.x >= grid_size :
                self.x = 0

            if self.y < 0:
                self.y = grid_size-1
            elif self.y >= grid_size :
                self.y = 0

            occupation_matrix[self.x,self.y] = 1
            self.rep_timeout -= 1
            #print(self.x,self.y,random_move)
            if self.rep_timeout <= 0:
                self.rep_timeout = 3
                return 1,occupation_matrix

            else:
                self.rep_timeout -= 1
                return 0,occupation_matrix
        else:
            #occupation_matrix[self.x,self.y] = 1
            return 0,occupation_matrix
        

    
def get_individuals(prey_n:int,shark_n:int,grid_size:int):
    fish_list = []
    shark_list = []


    occupation_grid_list = []

    for i in range(prey_n):
        x = np.random.randint(0,grid_size)
        y = np.random.randint(0,grid_size)
        while True:
            if [x,y] not in occupation_grid_list:  
                fish_list.append(prey(x,y,np.random.randint(0,4)))
                #occupation_grid_list.append([x,y])
                break
            else:
                x = np.random.randint(0,grid_size)
                y = np.random.randint(0,grid_size)


    for i in range(shark_n):
        x = np.random.randint(0,grid_size)
        y = np.random.randint(0,grid_size)
        while True:
            if [x,y] not in occupation_grid_list:
                shark_list.append(predator(x=x,y=y,rep_timeout=np.random.randint(0,20)))
                occupation_grid_list.append([x,y])
                break
            else:
                x = np.random.randint(0,grid_size)
                y = np.random.randint(0,grid_size)



    return fish_list,shark_list

def plot_fish(creature_list,occupation_matrix,iter_n):

    plt.figure()
    for fish in creature_list:
        if type(fish) is prey:
            occupation_matrix[fish.x,fish.y] = 1
            plt.plot(fish.x,fish.y,"b.")

        else:
            occupation_matrix[fish.x,fish.y] = 2
            plt.plot(fish.x,fish.y,"r.")
    plt.savefig(f"results/{iter_n}.png")
    plt.close()
    return occupation_matrix


fish_n = 300
shark_N = 50
grid_size = 40

fish_list, shark_list = get_individuals(fish_n,shark_N,grid_size)
occupation_matrix = np.zeros((grid_size,grid_size),dtype=int)

for fish in fish_list:
    occupation_matrix[fish.x,fish.y] = 1

for shark in shark_list:
    occupation_matrix[shark.x,shark.y] = 2


counter_list = [[fish_n,shark_N]]
for i in tqdm(range(1001)):
    #_ = plot_fish(creature_list,occupation_matrix,i)
    fish_counter = 0
    shark_counter = 0
    fish_to_kill = []
    a = 0
    survied_fish = []
    survied_sharks = []
    new_fish_list = fish_list.copy()
    new_shark_list = shark_list.copy()
    #print(len(fish_list))
    
    #fish movement and reproduction
    for j,fish in enumerate(fish_list):
        tmp_x, tmp_y = fish.x,fish.y
        
        # status = 0 just move, status 1 reproduction
        status,occupation_matrix = fish.move(occupation_matrix)
        
        if status == 1:
            new_fish_list.append(prey(tmp_x,tmp_y))
            occupation_matrix[tmp_x,tmp_y] = 1



    
    fish_list  = new_fish_list
    
    # shark movement, eating and reproduction
    for j,shark in enumerate(shark_list):
        tmp_x, tmp_y = shark.x,shark.y
        
        # status = 0 just move, status 1 reproduction, status 2 - eating
        status,occupation_matrix = shark.move(occupation_matrix)
        
        if status == 1:
            new_shark_list.append(predator(tmp_x,tmp_y))
            occupation_matrix[tmp_x,tmp_y] = 2
        elif status == 2:
            fish_to_kill.append([shark.x,shark.y])
        elif status == 3:
            new_shark_list.append(predator(tmp_x,tmp_y))
            occupation_matrix[tmp_x,tmp_y] = 2
            fish_to_kill.append([shark.x,shark.y])
    
    shark_list = new_shark_list

    # deleting dead fish and sharks

    for j,fish in enumerate(fish_list):
        if [fish.x,fish.y] in fish_to_kill:
            fish.alife = 0

    for j,shark in enumerate(shark_list):
        if shark.HP < 1:
            shark.alife = 0
            occupation_matrix[shark.x,shark.y] = 0



    #deleting
    for fish in fish_list:
        if fish.alife == 1:
            survied_fish.append(fish)

    for shark in shark_list:
        if shark.alife == 1:
            survied_sharks.append(shark)
            
    fish_list = survied_fish
    shark_list = survied_sharks

    #print(len(fish_list),len(shark_list),len(fish_list)+len(shark_list) )

    counter_list.append([len(fish_list),len(shark_list)])
    if i%10 == 0:
        plt.figure()
        plt.title(f"f = {len(fish_list)}, s = {len(shark_list)}")
        plt.imshow(occupation_matrix.T,vmin = 0,vmax=2)
        if i<10:
            zeros = "000"
        elif i < 100:
            zeros = "00"
        elif i < 1000:
            zeros = "0"
        plt.savefig(f"results_2/{zeros}{i}.png")
        plt.close()




counter_list = np.array(counter_list)
print(counter_list.shape)

plt.plot(counter_list.T[0])
plt.plot(counter_list.T[1])
plt.savefig("fish_in_time_40.png")
plt.close()

plt.plot(counter_list.T[1],counter_list.T[0])
plt.savefig("fish_vs_sharks_40.png")
plt.close()

np.save("dd_40.npy",counter_list)