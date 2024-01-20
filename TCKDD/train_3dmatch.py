#import os
#os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES']='0'

import torch
import numpy as np
from typing import Dict
from torch import optim
import matplotlib.pyplot as plt

from models.kpfcnn import KPFCNN
from engine.loss import HardestContrastiveLoss
from datasets.dataloader import get_dataloader
from datasets.match3d import ThreeDMatchTrainDataset


model = KPFCNN(1, 32, 64, 15, 0.075, 0.06, 'group_norm', 32)
model.train().cuda()


indoor = ThreeDMatchTrainDataset()
neighbor_limits = [37, 31, 34, 37, 35]
dloader, _ = get_dataloader(indoor, 0.03, 2.5, 5, 16, True) #, neighbor_limits
loss_fn = HardestContrastiveLoss(0.045, 0.1)
iterator = dloader.__iter__() # dataset iterator


# create optimizer
from munch import Munch
config = Munch()
config['optimizer'] = 'SGD'
config['momentum'] = 0.98
config['scheduler_gamma'] = 0.1**(1/80)
config['weight_decay'] = 1e-6
config['learning_rate'] = 1e-3
config['iterations'] = 40000
config['resume'] = 0
config['save'] = 1000
if config.optimizer == 'SGD':
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
elif config.optimizer == 'ADAM':
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=config.weight_decay,
    )
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.scheduler_gamma)
losses, desc_losses, pos_desc_losses, neg_desc_losses, det_losses = [], [], [], [], []


# resume: load weights and losses
if config.resume > 0:
    print('load model weights ...')
    weights_path = './checkpoints/3dmatch_kpfcnn_HCL64_%d.pth'%config.resume
    model.load_state_dict(torch.load(weights_path))
    print('load previous logs ...')
    
    with np.load('./logs/3dmatch_kpfcnn_HCL64.npz') as data:
        losses = data['loss'].tolist()
        det_losses = data['det_loss'].tolist()
        desc_losses = data['desc_loss'].tolist()
        pos_desc_losses = data['pos_desc_loss'].tolist()
        neg_desc_losses = data['neg_desc_loss'].tolist()
    for i in range(config.resume // config.save): scheduler.step()
    print(scheduler.get_last_lr())
    print("Done.")


for idx in range(config.resume, config.iterations):
    inputs = iterator.next()
    pcd0 = inputs['src_pcd']
    pcd1 = inputs['tar_pcd']
    trans = inputs['transform'].cuda()
    for k in pcd0.keys():
        if type(pcd0[k]) == list:
            pcd0[k] = [item.cuda() for item in pcd0[k]]
            pcd1[k] = [item.cuda() for item in pcd1[k]]
        else:
            pcd0[k] = pcd0[k].cuda()
            pcd1[k] = pcd1[k].cuda()
    
    desc0, saliency0 = model(pcd0)
    desc1, saliency1 = model(pcd1)
    result_dict: Dict[str, torch.Tensor] = loss_fn(
        trans, pcd0['points'][0], pcd1['points'][0], desc0, desc1, saliency0, saliency1
    )
    
    optimizer.zero_grad()
    result_dict['loss'].backward()
    optimizer.step()

    torch.cuda.empty_cache()
    result_dict = {k:v.cpu().item() for k,v in result_dict.items()}
    print("Step %06d:"%(idx+1), end=' ')
    for key, value in result_dict.items():
        print(key, "%.4f"%float(value), end='; ')
    print()
    
    pos_desc_losses.append(result_dict['pos_loss'])
    neg_desc_losses.append(result_dict['neg_loss'])
    desc_losses.append(result_dict['desc_loss'])
    det_losses.append(result_dict['det_loss'])
    losses.append(result_dict['loss'])
    
    if idx > 0 and (idx+1) % config.save == 0 and idx>25000:
        torch.save(model.state_dict(), './checkpoints/3dmatch_kpfcnn_HCL64_%d.pth'%(idx+1))
        if idx <= 40000: scheduler.step()


fig = plt.figure()
plt.subplot(2,2,1); plt.plot(np.array(det_losses)); plt.title('detection loss')
plt.subplot(2,2,2); plt.plot(np.array(desc_losses)); plt.title('descriptor loss')
plt.subplot(2,2,3); plt.plot(np.array(pos_desc_losses)); plt.title('positive descriptor loss')
plt.subplot(2,2,4); plt.plot(np.array(neg_desc_losses)); plt.title('negative descriptor loss')
plt.show()


loss_ema = losses.copy()
det_loss_ema = det_losses.copy()
desc_loss_ema = desc_losses.copy()
pos_desc_loss_ema = pos_desc_losses.copy()
neg_desc_loss_ema = neg_desc_losses.copy()

# disjoint sliding window average
window, yita = 5, 0.5
length = int(len(loss_ema)//window * window)
loss_ema = np.mean(np.array(loss_ema[:length]).reshape([-1, window]), axis=-1).tolist()
det_loss_ema = np.mean(np.array(det_loss_ema[:length]).reshape([-1, window]), axis=-1).tolist()
desc_loss_ema = np.mean(np.array(desc_loss_ema[:length]).reshape([-1, window]), axis=-1).tolist()
pos_desc_loss_ema = np.mean(np.array(pos_desc_loss_ema[:length]).reshape([-1, window]), axis=-1).tolist()
neg_desc_loss_ema = np.mean(np.array(neg_desc_loss_ema[:length]).reshape([-1, window]), axis=-1).tolist()

# exponential moving average
for i in range(1, len(loss_ema)):
    loss_ema[i] = yita * loss_ema[i-1] + (1.-yita) * loss_ema[i]
    det_loss_ema[i] = yita * det_loss_ema[i-1] + (1.-yita) * det_loss_ema[i]
    desc_loss_ema[i] = yita * desc_loss_ema[i-1] + (1.-yita) * desc_loss_ema[i]
    pos_desc_loss_ema[i] = yita * pos_desc_loss_ema[i-1] + (1.-yita) * pos_desc_loss_ema[i]
    neg_desc_loss_ema[i] = yita * neg_desc_loss_ema[i-1] + (1.-yita) * neg_desc_loss_ema[i]

fig = plt.figure()
plt.subplot(2,2,1); plt.plot(np.array(det_loss_ema)); plt.title('detection loss')
plt.subplot(2,2,2); plt.plot(np.array(desc_loss_ema)); plt.title('descriptor loss')
plt.subplot(2,2,3); plt.plot(np.array(pos_desc_loss_ema)); plt.title('positive descriptor loss')
plt.subplot(2,2,4); plt.plot(np.array(neg_desc_loss_ema)); plt.title('negative descriptor loss')
plt.show()


np.savez('./logs/3dmatch_kpfcnn_HCL64.npz',
         loss=np.array(losses),
         det_loss=np.array(det_losses),
         desc_loss=np.array(desc_losses),
         pos_desc_loss=np.array(pos_desc_losses),
         neg_desc_loss=np.array(neg_desc_losses),
        )