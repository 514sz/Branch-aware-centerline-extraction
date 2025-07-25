import numpy
import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
from Net import *
from utils import *

# Hyper-parameters
BATCH_SIZE = 32
LR = 0.0005  # learning rate
EPSILON = 0.7  # greedy policy
GAMMA = 0.9  # reward decay coefficient
TARGET_REPLACE_ITER = 100  # target replace frequency
MEMORY_CAPACITY = 100  # replay buffer size
N_ACTIONS = 500  # The action number

bound_thre = 15  # constrain explore region

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Tracker_Net(n_actions=N_ACTIONS), Tracker_Net(n_actions=N_ACTIONS)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, 8))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.loss_func.to(device)

    def choose_action(self, x):

        x = torch.unsqueeze(x, 0)
        y = torch.stack((x,x), dim=0)
        outputs = self.eval_net.forward(y)
        radi_value = outputs[:, -1, :, :, :]
        radi = radi_value.cpu().detach().numpy()[0]
        pr = abs(radi.item())
        if np.random.uniform() < EPSILON:  # greedy

            actions_value = outputs[:,:500,:,:,:]
            actions_value = actions_value.cpu()
            action = torch.max(actions_value, 1)[1].numpy()[0]
            action = action.item()

        else:  # randomly select action
            action = np.random.randint(0, N_ACTIONS)
        return action, pr

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # update replay buffer
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def step(self, s, a, pr):

        s_ = np.array([0, 0, 0])
        for i in range(direction_list.shape[0]):
            if a == i:
                s_ = s + direction_list[i]
                break

        dis1, g1 = mindis(s, vox_ref)
        dis2, g2 = mindis(s_, vox_ref)
        v1 = s_ - s
        v2 = vox_ref[g2] - vox_ref[g1]
        gr = radi_ref[g1]
        r = np.dot(v1, v2) + dis1 - dis2 - abs(pr - gr)
        done = False
        bound = region_constrain(vox_ref, bound_thre)
        if (s_[0] >= (bound[1] - 9) or s_[0] < (bound[0] + 9) or s_[1] >= (bound[3] - 9) or s_[1] < (bound[2] + 9) or
                s_[2] >= (bound[5] - 9) or s_[2] < (bound[4] - 9)):
            done = True

        return s_, r, done

    def learn(self):
        # target replaced by eval
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # batch sample
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = b_memory[:, :3]
        b_s = b_s.astype(np.int32)
        patches_b_s = []
        for i in range(b_s.shape[0]):
            patch = clip_patch(img_arr, b_s[i], gap=9)
            patches_b_s.append(patch)

        patches_b_s = np.asarray(patches_b_s, dtype=float)
        patches_b_s = torch.from_numpy(patches_b_s).float()
        patches_b_s = patches_b_s.unsqueeze(1)
        patches_b_s = patches_b_s.to(device)

        b_a = torch.LongTensor(b_memory[:, 3].astype(int))
        b_r = torch.FloatTensor(b_memory[:, 4])
        b_r = b_r.view(BATCH_SIZE, 1)
        b_r = b_r.to(device)

        b_s_ = b_memory[:, -3:]
        b_s_ = b_s_.astype(np.int32)
        patches_b_s_ = []
        for i in range(b_s_.shape[0]):
            patch = clip_patch(img_arr, b_s_[i], gap=9)
            patches_b_s_.append(patch)

        patches_b_s_ = np.asarray(patches_b_s_, dtype=float)

        patches_b_s_ = torch.from_numpy(patches_b_s_).float()
        patches_b_s_ = patches_b_s_.unsqueeze(1)
        patches_b_s_ = patches_b_s_.to(device)

        q_eval = self.eval_net(patches_b_s)
        q_eval = torch.squeeze(q_eval)
        b_a = b_a.view(BATCH_SIZE, 1)
        b_a = b_a.to(device)
        q_eval = q_eval.gather(1, b_a)
        q_next = self.target_net(patches_b_s_).detach()  # target needs no train
        q_next = torch.squeeze(q_next)
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


Tracker = DQN()
Tracker.eval_net.to(device)
Tracker.target_net.to(device)

print('\nCollecting experience...')

path_name = '/home/zyy/training/dataset'
for i in range(8):
    path_mhd = path_name + '0' + str(i) + '/image' + '0' + str(i) + '.mhd'
    img = sitk.ReadImage(path_mhd)
    img_arr = sitk.GetArrayFromImage(img)
    Spacing = img.GetSpacing()
    direction_list = create_actions(N_ACTIONS, dis=1.5, spacing=Spacing)

    for j in range(4):
        vox_radi_path = path_name + '0' + str(i) + '/vessel' + str(j) + '/vox_radi' + '.npy'
        vox_radi_ref = np.load(vox_radi_path)  # z,y,x,r
        vox_ref = vox_radi_ref[:, :3]  # z,y,x
        radi_ref = vox_radi_ref[:, 3]  # radius

        for i_episode in range(100000):
            s = np.array((vox_ref[0, 0], vox_ref[0, 1], vox_ref[0, 2]))  # z,y,x
            s = s.astype(int)

            ep_r = 0

            while True:
                patch = clip_patch(img_arr, s, gap=9)

                patch = patch.astype(float)
                patch = torch.from_numpy(patch).float()
                patch = patch.to(device)

                a, pr = Tracker.choose_action(patch)

                s_, r, did = Tracker.step(s, a, pr)

                if not did:
                    Tracker.store_transition(s, a, r, s_)

                ep_r += r
                if Tracker.memory_counter > MEMORY_CAPACITY:
                    Tracker.learn()
                    if did:
                        print('Ep: ', i_episode, '| Ep_r: ', round(ep_r, 2))

                if did:
                    break

                s = s_

torch.save(Tracker.eval_net.state_dict(), '/home/zyy/training/Tracker.pth')
