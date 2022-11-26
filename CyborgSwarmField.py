import numpy
import random
import math
import gym
import gym.spaces



pi = 3.14159265
a = 1
length = 1  #前進指令で進む距離  #オーバーするくらいがちょうどよい
d_theta = 0.01#0.058 リサージュ曲線について
fieldsize = 100
average_swarm = 4
bias = 1                    
radius = 0.5 #粒子の半径




class CyborgSwarmField(gym.core.Env):
    def __init__(self, n):
        self.number = n
        self.action_space = gym.spaces.Discrete(4) # 行動空間。
        high = numpy.array([n*0.1, n*0.1, 1.5, 1.5]) # 観測空間(state)の次元 (位置と速度の2次元) とそれらの最大値
        self.observation_space = gym.spaces.Box(low=-high, high=high) # 最小値は、最大値のマイナスがけ
        
        
        
    def statement(self, i):
        
        s1 = 0  #ポテンシャルエネルギー論に基づいた周辺の粒子のマクロ情報（x方向）
        s2 = 0  #ポテンシャルエネルギー論に基づいた周辺の粒子のマクロ情報（y方向）
        
        
        for j in range(self.number):
            if i != j:
                dx = self.list[i][0] - self.list[j][0]
                dy = self.list[i][1] - self.list[j][1]
                
                
                if dx != 0 or dy != 0:
                    s1 += a*dx*math.pow(dx*dx+dy*dy, -3/2)
                    s2 += a*dy*math.pow(dx*dx+dy*dy, -3/2)
                
                    
                
        disx = self.list[i][0] - self.ruuner[0]
        disy = self.list[i][1] - self.ruuner[1]
        
        
        
        
        return numpy.array([s1,s2, disx, disy])   
    
    
    
    def reward_to_target(self, i):
        
        disx = self.list[i][0] - self.ruuner[0]
        disy = self.list[i][1] - self.ruuner[1]
        
        dis = math.sqrt(disx*disx+disy*disy)  #runnerとの距離
        
        R_to_target = dis
        
        

        return R_to_target
    
    def find_min_distance(self,index):
        
        x_list = numpy.array([abs(t - self.list[index,0]) for t in self.list[:,0]])
        y_list = numpy.array([abs(t - self.list[index,1]) for t in self.list[:,1]])
        
        
       
        
        dis = numpy.array([math.sqrt(x_list[j]*x_list[j]+y_list[j]*y_list[j]) for j in range(self.number) if j != index])
        
        min_dis = numpy.min(dis)
        
        #print(min_dis)
        
        return min_dis
                          
        
        
    def step(self, a):
        
       
        
        
        
        #actionの種類
        # a = 0 =======> +1:x
        # a = 1 =======> -1:x
        # a = 2 =======> +1:y
        # a = 3 =======> -1:y
        
        #agentsの変化
        
        
        for index in range(self.number):
            if a[index] < 2:
                if a[index] == 0:
                    dx = length
                else:
                    dx = -length
                dy = 0
            else:
                dx = 0
                if a[index] == 2:
                    dy = length
                else:
                    dy = -length
                    
            #print(dx, dy)
            
            
                
            self.list[index][0] += dx
            self.list[index][0] = max(-fieldsize+1,  min(self.list[index][0], fieldsize-1))
            self.list[index][1] += dy
            self.list[index][1] = max(-fieldsize+1,  min(self.list[index][1], fieldsize-1))
            
        
        
   
            
        
         
        
        #runnerの移動（リサージュ曲線）
        self.count += 1
        
        theta = d_theta*self.count
        
        #dl = a*sqrt(4*cos^2(2theta)+9*cos^2(3theta))
        
        self.ruuner[0] = fieldsize * 0.7*math.sin(2*theta)
        self.ruuner[1] = fieldsize * 0.7*math.sin(3*theta)
        
        #print(2*theta, self.ruuner[1])
        
        
        
        
        #集合全体のマクロ的報酬swarm_rewardの計算
        min_dis_list = numpy.array([self.find_min_distance(i) for i in range(self.number)])
        variance = numpy.var(min_dis_list)/(fieldsize*fieldsize)  #集合距離の分散　　小さいほうが良い
        average = numpy.var(min_dis_list)/fieldsize  #集合距離の平均average_targetになるとよい
        average_reward = math.pow(average - average_swarm, 2) #小さいほうが良い
        
        
      
        
        
        
        swarm_reward = variance + average_reward/5
        
        
        #print(self.reward_to_target(5), variance, average_reward)       #これを順番をどのようにするか考えないといけないのか
        
        

        
        
        #biasによってどちらをどの程度優先するか設定することができる
        #print(self.reward_to_target(5))
    
    
        #ここも変えた
        g_list = numpy.array([-0.01 * (swarm_reward + bias*self.reward_to_target(i)) for i in range(self.number)])
        done = False
        
        
    
            
        #print(current_reward, self.last_reward)
        
        #print("list : ", self.list[0])
        #print("last_reward : ",self.last_reward[0])
    
        #if numpy.all(self.list == self.last_reward):
            #done = True
            #g_list += numpy.array([-1.0 for _ in range(self.number)])      #これはみそ
            
            
        self.last_reward = self.list.copy()
                
        
        
        #for i in range(self.number):
            #if self.find_min_distance(i) < 2*radius:
                #done = True                                  <==================  一旦考えないことにする。
                #g_list[i] -= 1.0
                #break
        s_list = numpy.array([self.statement(i) for i in range(self.number)])
        
        
        
        
        return s_list, g_list, done, {}
        
        
    
    
    #initialize location of agents
    def reset(self):
        self.list = numpy.array([[random.uniform(-fieldsize+2,fieldsize-2), 
                                  random.uniform(-fieldsize+2,fieldsize-2)] for _ in range(self.number)])
        self.last_reward = numpy.array([[0, 0] for _ in range(self.number)])
        self.ruuner = numpy.array([0.0,0.0])
        s_list = numpy.array([self.statement(i) for i in range(self.number)])
        self.count = 0  #計算ステップ
        
        return s_list
    
    
    
    
    

        
        

        
    
        
    
        
        
