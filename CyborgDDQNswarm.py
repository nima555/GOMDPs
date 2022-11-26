import random
from timeit import repeat
from turtle import color
import numpy as np
from collections import namedtuple, deque
from itertools import count
from matplotlib import pyplot
import matplotlib.animation as animation
from multiprocessing import Pool


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import DQN
import CyborgSwarmField

particle_number = 15
color_list = ["#800000" for i in range(particle_number)]
color_list.append("#1e90ff")

env = CyborgSwarmField.CyborgSwarmField(particle_number)


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))




class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    


    

    
    

#Network model
class DQN(nn.Module):

    def __init__(self, inputs, outputs):
       
        super(DQN, self).__init__()
        self.conv1 = nn.Linear(inputs, 16)
        self.conv2 = nn.Linear(16, 32)
        self.conv3 = nn.Linear(32, 32)
        self.conv4 = nn.Linear(32, 16)
        self.conv5 = nn.Linear(16, outputs)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        
        

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.sigmoid(self.conv5(x))
     
        
        return x
    
    







env.reset()




BATCH_SIZE = 512
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 5000
TARGET_UPDATE = 1
video_write = 50

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()



# Get number of actions from gym action space
n_actions = env.action_space.n
n_observe = env.observation_space.shape[0]


policy_net = DQN(n_observe, n_actions).to(device)
target_net = DQN(n_observe, n_actions).to(device)




target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(10000)



steps_done = 0 

def select_action(state, eps):
    
    sample = random.random()
    
    
    
    #eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        #math.exp(-1. * steps_done / EPS_DECAY)
    eps_threshold = eps
    
        
        
    #print(eps_threshold)
    
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            #print(state)
            #print(policy_net(state).max(1)[1])
            #print(policy_net(state).max(1)[1].view(1, 1))
            
            a = policy_net(state).max(1)[1].view(1, 1)
            #print(a)
            return a
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)





        

        
        
        
        
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    #print(batch.reward)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    #print(state_batch)
    #print(state_batch)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    #print(state_action_values, policy_net(state_batch))
    

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    #print(next_state_values)
    
    
    
    
    
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    #print(next_state_values[non_final_mask])
    #print(target_net(non_final_next_states).max(1))
    # Compute the expected Q values
    #print(reward_batch)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    #print(expected_state_action_values.unsqueeze(1))
    #print(state_action_values)

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad() # <===================== 研究の余地あり！！
    loss.backward()
    for param in policy_net.parameters():
        #print(param.grad)
        
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss


#def _update_plot(i, fig, im):
    
    
    
    
    
max_episode = 10000000




#fig = pyplot.figure()
#ax = fig.add_subplot(1,1,1)

#ax.set_xlim([-100, 100])
#ax.set_ylim([-100, 100])


#im = []





for i_episode in count():
    # Initialize the environment and state
    state = env.reset()
    state = torch.from_numpy(state.astype(np.float32)).to(device)
    
    
    
    if i_episode%video_write == 0:
        fig = pyplot.figure()
        rend_data = []
     
    #steps_done = 0
    
    for t in count():
        
        
        
    
        #im = []
            
        
        # Select and perform an action
        #action = select_action(state, 0.2)
        action = []
        
        
        steps_done += 1
        for i in range(particle_number):
            
            x = select_action(state[i].view(1,-1), 0.2)
            action.append(x)
        action = torch.cat(action, dim = 1) 
        
        

        
        
        next_state, reward, done, _ = env.step(np.array([action[0,q].item() for q in range(particle_number)]))
        #print(env.list[:,0], t)
        
        
        if i_episode%video_write == 0:
            im = pyplot.scatter(np.append(env.list[:,0], env.ruuner[0]), np.append(env.list[:,1], env.ruuner[1]), s=7, color = color_list)
            #im = pyplot.scatter(env.ruuner[0], env.ruuner[1], s=2.5, color = "#1e90ff")
            #print(len(env.list[:,0]))
            #pyplot.show()
            rend_data.append([im])
            pass
        
        
        #print(np.shape(rend_data))
        
        
        reward = torch.tensor([reward], device=device)
        #print(type(next_state))
        next_state = torch.from_numpy(next_state.astype(np.float32)).to(device)
        

        # Observe new state
        
        #print(reward[0,i].view(1))

        # Store the transition in memory
        for i in range(particle_number):
            memory.push(state[i].view(1,-1), action[0,i].view(1,-1), next_state[i].view(1,-1), reward[0,i].view(1))

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        loss = optimize_model()

        
        if t >= 500:
            print("loss :", loss.item(), "      episodes :", t, "       all_steps :", steps_done, "     reward :", torch.mean(reward).item())
            break
        
        if done:
            
            if loss != None:
                print("loss :", loss.item(), "      episodes :", t, "       all_steps :", steps_done, "     reward :", torch.mean(reward).item())
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        
    if i_episode % 200 == 0:
        torch.save(policy_net.state_dict(), "model_swarm_3.pth")
        print("Torch model saved")
        
    if i_episode%video_write == 0:
        ani = animation.ArtistAnimation(fig, rend_data, interval = 50, repeat = True)
        ani.save("swarm_video_3.gif", writer="imagemagick")
        print("Movie saved")
        fig.clear()
        pyplot.close()
        #pyplot.save()
        
    if steps_done > max_episode:
        break
    

print('Complete')




















"""
def select_action_test(state_a):
    #print(policy_net(state).max(1)[1].view(1, 1))
    return policy_net(state_a).max(1)[1].view(1, 1)




    
for i in range(5):
    
    

    test_data = []


    state = env.reset()
    state = torch.from_numpy(state.astype(np.float32)).to(device).view(1,-1)
    test_data.append(state[0,0].cpu().item())


    for t in count():
        
        
        #print(state)
        action = select_action(state, 0)
        next_state, reward, done, _ = env.step(action.item())
        #print(next_state, action)
        reward = torch.tensor([reward], device=device)
        next_state = torch.from_numpy(next_state.astype(np.float32)).to(device).view(1,-1)
        state = next_state
        

        # Perform one step of the optimization (on the policy network)
        
        test_data.append(state[0,0].cpu().item())

        if t >= 5000:
            print(t)
            break
        
        if done:
            
            print(t)
            break






    print(test_data)





    x = [i for i in range(len(test_data))]

    #print(type(test_data[5]))

    pyplot.plot(x,test_data)
    pyplot.show()


"""
