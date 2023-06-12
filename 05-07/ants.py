import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

class ant:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.direction = np.random.randint(0,8)
        self.food = False
        self.strength = 1
        self.counter = 300
        
    
    def move(self,grid,grid_f1,grid_f2,summary):
        if self.counter < 0:
            self.direction = np.random.randint(0,8)
            self.counter = 300
        elif self.food == True:
            self.direction = self.check_surroundings(grid,grid_f1)
        else:
            self.direction = self.check_surroundings(grid,grid_f2)

        self.x = self.x % 80
        self.y = self.y % 80
        self.strength *= 0.99
        if self.food == False:
            grid_f1[self.x,self.y] += self.strength
        else: 
            grid_f2[self.x,self.y] += self.strength
        self.x += directions[self.direction,0]
        self.y += directions[self.direction,1]
        #przenosi do nowej kratki
        self.x = self.x % 80
        self.y = self.y % 80

        if grid[self.x,self.y] == 1 and self.food == False:
            self.direction += 4
            self.direction = self.direction % 8
            self.food = True
            grid[self.x,self.y] = 0
            self.counter = 300
            self.strength = 1
        elif grid[self.x,self.y] == 2  and self.food == True:
            self.direction += 4
            self.direction = self.direction % 8
            self.food = False
            summary += 1
            self.strength = 1
            self.counter = 300
            print("new food coleceted", summary)

        self.counter -=1
        return summary
    def update_feromons(self,grid_type):
        grid_type[self.x,self.y] = 1

    def check_surroundings(self,grid,grid_f):
        tmp = np.array([-1,0,1])
        dir_to_check = tmp + self.direction
        dir_to_check = dir_to_check % 8
        
        if self.food == False:
            searching = 1
        else: searching = 2

        good_points = []
        good_points_weights = []
        for j,dir in enumerate(dir_to_check):
            new_cords = (np.array([self.x,self.y]) + directions[dir]) % 80
            if grid[new_cords[0],new_cords[1]] == searching:
                new_direction = dir
                return new_direction
            elif grid[new_cords[0],new_cords[1]]!=1 and grid[new_cords[0],new_cords[1]]!=3:
                good_points_weights.append(grid_f[new_cords[0],new_cords[1]])
                good_points.append(dir)

        good_points_weights = np.array(good_points_weights)

        #good_points_weights = (good_points_weights + 2)**2
        if np.sum(good_points_weights) > 0:
            dir = np.random.choice(good_points,p = good_points_weights/np.sum(good_points_weights))
            return dir
        elif np.sum(good_points_weights) == 0 and len(good_points)>0:
            new_direction = np.random.choice(good_points)
        else: 
            new_direction = self.direction + 4
            new_direction %= 8
        return new_direction
    
directions = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]])
grid = np.zeros((80,80))
grid_f1 = np.zeros((80,80)) # feromonte 1
grid_f2 = np.zeros((80,80)) # feromnte 2

start_x, start_y = 40,20




grid[start_x, start_y] = 2
grid[10:70, 50:75] = 1
grid[35:45, 60:75] = 1
grid[0:80, 0:5] = 3 
grid[15:65, 45:50] = 3 
ant_list = []
list_to_save = []
summary = 0
list_to_save = [0]
for i in range(10_000):
    if len(ant_list) < 15:
        ant_list.append(ant(start_x, start_y))
    for each_ant in ant_list:
        summary = each_ant.move(grid,grid_f1,grid_f2,summary)
        # print(each_ant.direction)


    grid_f1 *= 0.99
    grid_f2 *= 0.99
    grid_f1 = gaussian_filter(grid_f1,0.05)
    grid_f2 = gaussian_filter(grid_f2,0.05)

    grid_f1 = np.where(grid!=0, 0,grid_f1)
    grid_f2 = np.where(grid!=0, 0,grid_f2)
    list_to_save.append(summary)

    if i % 10 == 0:
        plt.figure()
        ax = plt.gca()
        ax.matshow(grid, cmap='Blues',origin='lower')
        ax.matshow(grid_f1,vmax=1,alpha=0.5,cmap='Oranges',origin='lower')
        ax.matshow(grid_f2,vmax=1,alpha=0.5,cmap='Greens',origin='lower')
        for each_ant in ant_list:
            if each_ant.food == False:
                plt.plot(each_ant.y,each_ant.x,'r.')
            else:
                plt.plot(each_ant.y,each_ant.x,'g.')

        if i < 10:
            title = f"0000{i}"
        elif i < 100:
            title = f"000{i}"
        elif i < 1000:
            title = f"00{i}"
        elif i < 10000:
            title = f"0{i}"
        plt.savefig(f"results_3/{title}_1.png")
        plt.close()

    if summary >= np.sum(np.where(grid==1)):
        print(f"wszystko zjedzone w {i} iteracjach")
        break
        
arr = np.array(list_to_save)
plt.figure()
plt.plot(arr)
plt.savefig("summary.png")
plt.close()