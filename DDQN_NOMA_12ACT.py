import math
import random
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import rl_utils
import torch
import torch.nn
import torch.nn.functional as F
from tqdm import tqdm
import netron
import torch.onnx


#设计第一条uav飞行路线——基于信号收发的合速率

#无人机飞行过程中满足下列参数：
T=30
l=50
Tl=T/l
vmax=25
dmax=vmax*Tl
N0=105
beta=21
G0=9
G1=9
P0=10
h=100
#定义收缩尺度：R
s=0.2
T0 = Tl / 2
T1 = Tl / 2
#上下行链路时间均分

def harvest(d):
    return G0*G1*beta/(math.pow((d),2))

def dist(x1,y1, x2, y2, h):
    return math.sqrt(math.pow(x1-x2,2)+math.pow(y1-y2,2)+pow(h,2))
def rsum(x1,y1,x2,y2,x3,y3):
   return (T1/math.log(2))*math.log(1+G0*G1*T0*P0*(math.pow(harvest(dist(x1,y1,x2,y2,h)),2)+
                                                   math.pow(harvest(dist(x1,y1,x3,y3,h)),2 ))/(T1*N0))
destination=[300,300]
class uav:
    def __init__(self):
        self.x=70
        self.y=70
        self.track=deque()
        self.direction = ['w', 'a', 's', 'd','wa1','wd1','sa1','sd1','wa2','wd2','sa2','sd2']
        self.direct = 'w'
        self.done=False
        self.track_num=0
        self.speed=vmax*Tl
    def move(self,direct):
           self.track.append((self.x, self.y))
           if direct=='w':
             self.direct = direct
             self.x=self.x
             self.y=self.y+self.speed
           elif direct=='a':
             self.direct = direct
             self.x=self.x-self.speed
             self.y=self.y
           elif direct == 's':
             self.direct = direct
             self.x=self.x
             self.y=self.y-self.speed
           elif direct == 'd':
             self.direct = direct
             self.x=self.x+self.speed
             self.y=self.y
           elif direct =='wa1':
               self.direct = direct
               self.x = self.x - self.speed*math.sqrt(3)/2
               self.y = self.y + self.speed/2
           elif direct == 'wd1':
               self.direct = direct
               self.x = self.x + self.speed * math.sqrt(3) / 2
               self.y = self.y + self.speed / 2
           elif direct == 'sd1':
               self.direct = direct
               self.x = self.x + self.speed * math.sqrt(3) / 2
               self.y = self.y - self.speed / 2
           elif direct == 'sa1':
               self.direct = direct
               self.x = self.x - self.speed * math.sqrt(3) / 2
               self.y = self.y - self.speed /2
           elif direct == 'wa2':
               self.direct = direct
               self.x = self.x - self.speed / 2
               self.y = self.y + self.speed * math.sqrt(3) / 2
           elif direct == 'wd2':
               self.direct = direct
               self.x = self.x + self.speed / 2
               self.y = self.y + self.speed * math.sqrt(3) / 2
           elif direct == 'sd2':
               self.direct = direct
               self.x = self.x + self.speed/ 2
               self.y = self.y - self.speed * math.sqrt(3) / 2
           elif direct == 'sa2':
               self.direct = direct
               self.x = self.x - self.speed / 2
               self.y = self.y - self.speed * math.sqrt(3) / 2

    def reset(self):
        self.track.clear()
        self.track_num=0
        self.x = 70
        self.y = 70
        self.track.append((self.x, self.y))
        self.done=False

class noma_env():
    def __init__(self,uav,terminals,step_num):
        self.uav=uav
        self.uav.track.clear()
        self.uav.track.append((70,70))
        self.terminals=terminals
        self.step_num=step_num
        self.Rsum=deque()
        self.Rsum.append(self.R_sum())
        self.state=[0 for _ in range(13)]

  #  由分簇优化可以得到26 / 48 / 13 / 57, 由此分簇为c—noma最优分簇
    def R_sum(self):
            cost= rsum(self.uav.x,self.uav.y,self.terminals[0][0],self.terminals[0][1],self.terminals[5][0],
            self.terminals[5][1])+rsum(self.uav.x,self.uav.y,self.terminals[3][0],self.terminals[3][1],self.terminals[1][0],
            self.terminals[1][1])+rsum(self.uav.x,self.uav.y,self.terminals[6][0],self.terminals[6][1],self.terminals[4][0],
            self.terminals[4][1])+rsum(self.uav.x,self.uav.y,self.terminals[7][0],self.terminals[7][1],self.terminals[2][0],
            self.terminals[2][1])
            self.Rsum.append(cost)
    def newstate(self):
        self.state = [0 for _ in range(13)]
        for i in range(len(self.terminals)):
            self.state[i]=dist(self.uav.x,self.uav.y,self.terminals[i][0],self.terminals[i][1],h)
        self.state[8]=self.uav.track_num
        self.state[9]=self.uav.x
        self.state[10]=self.uav.y
        self.state[11]=300-self.uav.x
        self.state[12]=300-self.uav.y
        return self.state

    def runonestep(self, action):
        self.uav.track_num = self.uav.track_num + 1
        self.uav.move(self.uav.direction[action])
        self.R_sum()
        newstate = self.newstate()
        done=self.uav.done
        if self.uav.track_num > self.step_num:
            self.uav.done = True
        else :uav.done = False
        rewards = self.Rsum[-1]
        #print(rewards)
        if(rewards>0.06):
         rewards *= 2000
        for j in range(len(self.uav.track) - 1):
            if dist(self.uav.x, self.uav.y, self.uav.track[j][0], self.uav.track[j][1], 0) < self.uav.speed / 1.5:
                rewards = -1000
                break
        if(self.uav.x<10):
            rewards=-1000
        elif(self.uav.x>300):
            rewards = -1000
        elif (self.uav.y< 10):
            rewards =-1000
        elif (self.uav.y > 300):
            rewards =-1000
        else: rewards=rewards
            #self.uav.done=True
        return newstate, rewards, done

    def reset(self):
        self.uav.reset()
        self.Rsum.clear()
        self.state = [0 for _ in range(13)]
        return self.state

class qnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim1,hidden_dim2, action_dim):
        super(qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim1)
        self.fc2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = torch.nn.Linear(hidden_dim2, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

class DQN:
    def __init__(self, state_dim,
                 hidden_dim1,
                 hidden_dim2,
                 action_dim,
                 learning_rate,
                 gamma,
                 epsilon,
                 target_update):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.lr = learning_rate
        self.qnet = qnet(state_dim, hidden_dim1,hidden_dim2, action_dim)
        # self.qnet.weight.data.normal_(1.0, 0.02)
        self.target_qnet = qnet(state_dim, hidden_dim1,hidden_dim2, action_dim)
        # self.optimizer = torch.optim.SGD(self.qnet.parameters(), lr=learning_rate,momentum=0.09)
        self.optimizer = torch.optim.Adagrad(self.qnet.parameters(), lr=learning_rate)

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float)
            action = self.qnet(state).argmax().item()
        return action

    def max_q_value(self, state):
        state = torch.tensor([state], dtype=torch.float)
        return self.qnet(state).max().item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1)
        q_values = self.qnet(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值

        max_action = self.qnet(next_states).max(1)[1].view(-1, 1)
        max_next_q_values = self.target_qnet(next_states).gather(1, max_action)

        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_qnet.load_state_dict(
                self.qnet.state_dict())  # 更新目标网络
        self.count += 1


def train_DQN(agent, env, num_episodes, replay_buffer, minimal_size,
              batch_size):
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    for i in range(10):
        env.reset()
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(
                        state) * 0.005 + max_q_value * 0.995  # 平滑处理
                    max_q_value_list.append(max_q_value)  # 保存每个状态的最大Q值
                    next_state, reward, done = env.runonestep(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(
                            batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:]),
                    })
                pbar.update(1)
        env.reset()
        state = env.newstate()
        while (env.uav.done == False):
            state = torch.tensor([state], dtype=torch.float)
            if np.random.random() < epsilon:
                action = np.random.randint(action_dim)
            else:
                action = agent.target_qnet(state).argmax().item()
            next_state, reward, done = env.runonestep(action)
            state = next_state
        print(env.uav.track)
        print('the uplink achievable sum rate=', sum(env.Rsum) / len(env.Rsum))
        x = list()
        y = list()
        for i in range(len(env.uav.track)):
            x.append(env.uav.track[i][0])
            y.append(env.uav.track[i][1])
        fig, ax = plt.subplots()
        ax.set(title="Trajectory Plot", xlabel="X", ylabel="Y")
        plt.grid()
        plt.gca().set_aspect(1)
        ax.plot(x, y)
        plt.scatter(x, y)
        for i in range(len(terminals)):
            plt.scatter(terminals[i][0], terminals[i][1])
        plt.show()

    return return_list, max_q_value_list


def save_model(save_path, iteration, optimizer, model):
    torch.save({'iteration': iteration,
                'optimizer_dict': optimizer.state_dict(),
                'model_dict': model.state_dict()},
               save_path)
    print("model save success")


def load_model(save_name, optimizer, model):
    model_data = torch.load(save_name)
    model.load_state_dict(model_data['model_dict'])
    optimizer.load_state_dict(model_data['optimizer_dict'])
    print("model load success")

lr = 1e-2
num_episodes = 2000


path = 'noma_uav_12move.pkl'
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
gamma = 0.98
epsilon = 0.01
target_update = 100
buffer_size = 5000
minimal_size = 1000
batch_size = 1000
hidden_dim1 = 128
hidden_dim2 = 128
uav=uav()
terminals=[[125,32],[55,54],[81,170],[153,223],[190,61],[252,109],[275,197],[245,266]]

step_num=l
env=noma_env(uav,terminals,step_num)
state_dim = 13
action_dim = 12
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
agent = DQN(state_dim, hidden_dim1,hidden_dim2, action_dim, lr, gamma, epsilon,
            target_update)

load_model(path, agent.optimizer, agent.qnet)
load_model(path, agent.optimizer, agent.target_qnet)

train =False

if train:
    return_list, max_q_value_list = train_DQN(agent, env, num_episodes,
                                              replay_buffer, minimal_size,
                                              batch_size)
else:
    return_list = []
    max_q_value_list = []

if train:

    episodes_list = list(range(len(return_list)))
    mv_return = rl_utils.moving_average(return_list, 5)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format('greedy_snake'))
    plt.show()

    frames_list = list(range(len(max_q_value_list)))
    plt.plot(frames_list, max_q_value_list)
    plt.axhline(0, c='orange', ls='--')
    plt.axhline(10, c='red', ls='--')
    plt.xlabel('Frames')
    plt.ylabel('Q value')
    plt.title('DQN on {}'.format('greedy_snake'))
    plt.show()
    save_model(path, num_episodes, agent.optimizer, agent.target_qnet)

env.reset()
state=env.newstate()
while(env.uav.done==False):
    state = torch.tensor([state], dtype=torch.float)
    if np.random.random() < epsilon:
        action = np.random.randint(action_dim)
    else:
        action = agent.target_qnet(state).argmax().item()
    next_state, reward, done = env.runonestep(action)
    state = next_state
print(env.uav.track)
print('the uplink achievable sum rate=',sum(env.Rsum)/len(env.Rsum))
x=list()
y=list()
for i in range(len(env.uav.track)):
    x.append(env.uav.track[i][0])
    y.append(env.uav.track[i][1])
fig, ax = plt.subplots()
ax.set(title="Trajectory Plot", xlabel="X", ylabel="Y")
plt.grid()
plt.gca().set_aspect(1)
ax.plot(x, y)
plt.scatter(x, y)
for i in range(len(terminals)):
    plt.scatter(terminals[i][0],terminals[i][1])
plt.show()

"""
env.reset()
state = env.newstate()
state = torch.tensor([state], dtype=torch.float)
onnx_path = "onnx_model2_name.onnx"
torch.onnx.export(agent.target_qnet, state, onnx_path)

netron.start(onnx_path)
"""