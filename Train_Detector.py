import torch.optim
from torch import nn
from torch.utils.data import DataLoader
from Net import *
from MyDataset import MyDataset

# Hyper Parameters
BATCH_SIZE = 32
LR = 5*1e-4  # learning rate
epoch = 1000
r_factor = 10

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

path_name = '/home/zyy/training/dataset'
train_dataset = MyDataset('/home/zyy/training/dataset00/vessel0/patch', '/home/zyy/training/dataset00/vessel0/detector_label')
for i in range(8):
    for j in range(4):
        if j == 0:
            continue
        else:
            patch_dir = path_name + '0' + str(i) + '/vessel' + str(j) + '/patch'
            label_dir = path_name + '0' + str(i) + '/vessel' + str(j) + '/detector_label'
            dataset = MyDataset(patch_dir, label_dir)
            train_dataset += dataset

data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


detector = Detector_Net()
detector.load_state_dict(torch.load(r'/home/zyy/training/Detector.pth'))
detector.to(device)
# detector.load_state_dict(torch.load(r'/home/zyy\training\Detector.pth'))
loss_fn1 = nn.BCEWithLogitsLoss()
loss_fn2 = nn.BCEWithLogitsLoss()
loss_fn3 = nn.MSELoss()
loss_fn1.to(device)
loss_fn2.to(device)
loss_fn3.to(device)

optimizer = torch.optim.Adam(detector.parameters(), lr=LR)

for i in range(epoch):
    total_train_loss = 0
    print('第{}轮训练'.format(i))
    detector.train()
    # step = 0
    for data in data_loader:
        patches, labels = data
        patches, labels = patches.to(device), labels.to(device)
        patches = torch.unsqueeze(patches, 1)
        patches = patches.type(torch.float32)
        labels = labels.type(torch.float32)
        outputs = detector(patches)
        outputs = torch.squeeze(outputs)
        loss1 = loss_fn1(outputs[:, 0], labels[:, 0])
        loss2 = loss_fn2(outputs[:, 1], labels[:, 1])
        loss3 = loss_fn3(outputs[:, 2], labels[:, 2])
        loss = loss1 + loss2 + r_factor*loss3
        # loss = loss.type(torch.float)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss
        # print('第{}步'.format(step))
        # step+=1
    print('第{}轮总损失{}'.format(i,total_train_loss))

torch.save(detector.state_dict(), r'/home/zyy/training/Detector.pth')


