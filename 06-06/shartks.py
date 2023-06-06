import numpy as np
from matplotlib import pyplot as plt
import random

# tracking them as list of individuals 
# operation as numpy matrices


class individuals():
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.birth_counter = 0
        self.rep_timeout = 10
        self.alife = 1
    
    

class predator(individuals):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.rep_timeout = 20
        self.HP = 3
    def __str__(self):
        return f"shark {self.x},{self.y}"
    
    def move(self,occupation_matrix):
        grid_size = 40
        e = np.array(
        [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
    )
        possible_move = np.zeros(9,dtype=int)
        for i in range(9):

            possible_move[i] = occupation_matrix[self.x,self.y] - np.roll(np.roll(occupation_matrix,e[i][0],axis=0),e[i,1],axis=1)[self.x,self.y]


        if np.any(possible_move==1):
            move = np.random.choice(possible_move[possible_move==1])
            eating = 1
        
        elif np.any(possible_move>1):
            move = np.random.choice(possible_move[possible_move>1])
            eating = 0

        else:
            eating = 0
            return eating,occupation_matrix

        occupation_matrix[self.x,self.y] = 0
        
        #boundary condition
        self.x += e[move,0]
        self.y += e[move,1]
        try:
            occupation_matrix[self.x,self.y] = 2
        except IndexError:
            if self.x < 0:
                self.x = grid_size-1
            elif self.x >= grid_size :
                self.x = 0

            if self.y < 0:
                self.y = grid_size-1
            elif self.y >= grid_size :
                self.y = 0

            occupation_matrix[self.x,self.y] = 2
    
        return eating,occupation_matrix
class prey(individuals):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.rep_timeout = 3
    def __str__(self):
        return f"fish {self.x},{self.y}"
    
    def move(self,occupation_matrix):
        grid_size = 40
        e = np.array(
        [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
    )
        possible_move = np.zeros(9,dtype=int)
        for i in range(9):

            possible_move[i] = occupation_matrix[self.x,self.y] - np.roll(np.roll(occupation_matrix,e[i][0],axis=0),e[i,1],axis=1)[self.x,self.y]


        if np.any(possible_move>0):
            random_move = np.random.choice(np.where(possible_move>0)[0])
        else:
            return 0,occupation_matrix
        #print(random_move)
        # x,y update
        occupation_matrix[self.x,self.y] = 0
        #boundary condition
        self.x += e[random_move,0]
        self.y += e[random_move,1]
        try:
            occupation_matrix[self.x,self.y] = 1
        except IndexError:
            if self.x < 0:
                self.x = grid_size-1
            elif self.x >= grid_size :
                self.x = 0

            if self.y < 0:
                self.y = grid_size-1
            elif self.y >= grid_size :
                self.y = 0

            occupation_matrix[self.x,self.y] = 1
        
        return 0,occupation_matrix

    



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



def get_individuals(prey_n:int,shark_n:int,grid_size:int):
    creature_list = []
    #creature_list = []


    occupation_grid_list = []

    for i in range(prey_n):
        x = np.random.randint(0,grid_size)
        y = np.random.randint(0,grid_size)
        while True:
            if [x,y] not in occupation_grid_list:  
                creature_list.append(prey(x,y))
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
                creature_list.append(predator(x,y))
                occupation_grid_list.append([x,y])
                break
            else:
                x = np.random.randint(0,grid_size)
                y = np.random.randint(0,grid_size)



    return creature_list


fish_n = 300
shark_N = 10
grid_size = 40

creature_list = get_individuals(fish_n,shark_N,grid_size)
occupation_matrix = np.zeros((40,40),dtype=int)

    # initialization is working!


#prey_list.sort(key=lambda x: x.x, reverse=True)
occupation_matrix = plot_fish(creature_list,occupation_matrix,0)
plt.figure()
plt.imshow(occupation_matrix)
plt.savefig("aa")

for i in range(100):
    #_ = plot_fish(creature_list,occupation_matrix,i)

    fish_to_kill = []
    a = 0

    survied = []
    for fish in creature_list:
        status,occupation_matrix = fish.move(occupation_matrix)

        # zjadanie
        if status == 1:
            #print("eating",fish,i)
            fish_to_kill.append([fish.x,fish.y])

    
    for fish in creature_list:
        if type(fish) is prey and [fish.x,fish.y] in fish_to_kill:
            fish.alife = 0
            #print("eating",fish,i)
            #occupation_matrix[fish.x,fish.y] = 0

    

    fish_counter = 0
    shark_counter = 0
    for fish in creature_list:
        if fish.alife == 1:
            if type(fish) == prey:
                fish_counter+=1

            else:
                shark_counter+=1
            survied.append(fish)


    creature_list=survied
    print(fish_counter,shark_counter)
    #print(len(creature_list))
    #print(np.sum(occupation_matrix))

    if i %10 == 0:
        plt.figure()
        plt.imshow(occupation_matrix.T)
        plt.savefig(f"results/{i}.png")
        plt.close()
    
    

    
