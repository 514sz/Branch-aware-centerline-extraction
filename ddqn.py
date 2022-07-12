import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import SimpleITK as sitk

# Hyper Parameters
BATCH_SIZE = 64
LR = 0.0005                  # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000 # experience replay
N_ACTIONS = 6 # action number
GAP = 9 # patch size
device = torch.device('cuda:2')

path_mhd = r'/Share6/zyy/dataset00/image00.mhd'
image = sitk.ReadImage(path_mhd)
img_arr = sitk.GetArrayFromImage(image) # ct image array
vox_radi_ref = np.load(r'/Share6/zyy/dataset00/vessel0/vox_radi_label.npy')
vox_ref = vox_radi_ref[:, :3] # position reference of centerline points(x,y,z)

def clip_patch(img_arr, s):
    patch = img_arr[int(s[0])-GAP:int(s[0])+GAP+1, int(s[1])-GAP:int(s[1])+GAP+1, int(s[2])-GAP:int(s[2])+GAP+1]
    return patch

def mindis(p, set):
    index = 0
    min_dis = np.sqrt(np.sum((p - set[0, :]) ** 2))
    for i in range(set.shape[0]):
        dis = np.sqrt(np.sum((p - set[i, :]) ** 2))
        if dis <= min_dis:
            min_dis = dis
            index = i
    return min_dis, index


class Net(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, dilation=1),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, dilation=1),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, dilation=2),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, dilation=4),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, dilation=1),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=1, stride=1, dilation=1),
            nn.ReLU()
        )

        self.out = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=6, kernel_size=1, stride=1, dilation=1),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        actions_value = self.out(x6)
        return actions_value

def init_weights(m):
    if isinstance(m, nn.Conv3d):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net().to(device), Net().to(device)
        self.eval_net.apply(init_weights)
        self.target_net.apply(init_weights)

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, 8))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):

        x = torch.unsqueeze(torch.unsqueeze(x, 0), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net(x)
            actions_value = actions_value.to('cpu')
            action = torch.max(actions_value, 1)[1].numpy() # return the argmax index
            action = action[0][0][0][0]
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            # action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def step(self, s, a):
        s_ = np.array((0, 0, 0))
        done = False
        if a == 0:
            s_ = np.array([s[0] + 2, s[1], s[2]])
        elif a == 1:
            s_ = np.array([s[0] - 2, s[1], s[2]])
        elif a == 2:
            s_ = np.array([s[0], s[1] - 2, s[2]])
        elif a == 3:
            s_ = np.array([s[0], s[1] + 2, s[2]])
        elif a == 4:
            s_ = np.array([s[0], s[1], s[2] - 2])
        elif a == 5:
            s_ = np.array([s[0], s[1], s[2] + 2])

        dis1, g1 = mindis(s, vox_ref)
        dis2, g2 = mindis(s_, vox_ref)
        v1 = s_ - s
        v2 = vox_ref[g2] - vox_ref[g1]
        r = np.dot(v1, v2)+dis1-dis2

        if s_[0]+9 >= img_arr.shape[0] or s_[0]-9<0 or s_[1]+9 >= img_arr.shape[1] or s_[1]-9<0 or s_[2]+9 >= img_arr.shape[2] or s_[2]-9<0:
            done = True

        return s_, r, done



    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = b_memory[:, :3] # numpy
        b_a = torch.LongTensor(b_memory[:, 3].astype(int)).to(device)
        b_r = torch.FloatTensor(b_memory[:, 4]).to(device)
        b_s_ = b_memory[:, -3:] # numpy

        # q_eval w.r.t the action in experience
        p_b_s = np.zeros((BATCH_SIZE,2*GAP+1,2*GAP+1,2*GAP+1))
        for i in range(p_b_s.shape[0]):
            p_b_s[i] = clip_patch(img_arr, b_s[i])
        p_b_s = torch.from_numpy(p_b_s.astype(np.float32))
        p_b_s = torch.unsqueeze(p_b_s, 1) # [batch, channel, z,y,x]
        p_b_s = p_b_s.to(device)

        q_eval = self.eval_net(p_b_s).squeeze() # shape (batch, N_ACTIONS)
        b_a = b_a.unsqueeze(1) # [batch, 1]
        q_eval = q_eval.gather(1, b_a)

        p_b_s_ = np.zeros((BATCH_SIZE, 2 * GAP + 1, 2 * GAP + 1, 2 * GAP + 1))
        for i in range(p_b_s_.shape[0]):
            p_b_s_[i] = clip_patch(img_arr, b_s_[i])
        p_b_s_ = torch.from_numpy(p_b_s_.astype(np.float32))
        p_b_s_ = torch.unsqueeze(p_b_s_, 1)
        p_b_s_ = p_b_s_.to(device)

        q_next = self.target_net(p_b_s_).detach()     # detach from graph, don't backpropagate
        q_next = q_next.squeeze()
        b_r = b_r.unsqueeze(1)
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()


for i_episode in range(40000):
    s = np.array((vox_ref[0, 2], vox_ref[0, 1], vox_ref[0, 0])) # z,y,x
    ep_r = 0
    while True:
        patch = clip_patch(img_arr, s)
        patch = patch.astype(int)
        patch = torch.from_numpy(patch).float()
        patch = patch.to(device)

        # choose action
        a = dqn.choose_action(patch)

        # take action
        s_, r, done = dqn.step(s, a)

        print('a')
        # store transition
        if not done:
            dqn.store_transition(s, a, r, s_)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep: ', i_episode, '| Ep_r: ', round(ep_r, 2))

        if done:
            break
        s = s_

torch.save(dqn.eval_net.state_dict(), 'eval_net.pkl')


