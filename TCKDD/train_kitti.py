#import os
#os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES']='0'
import time
import numpy as np
from munch import Munch
import matplotlib.pyplot as plt
from models.kpfcnn import KPFCNN
from datasets.kitti import KITTIDataset
from trainer import HardestContrastiveLoss
from datasets.dataloader import get_dataloader

import torch
from torch import optim


model = KPFCNN(1, 32, 64, 15, 0.9, 0.6, 'group_norm', 32)
model.train().cuda()

kitti = KITTIDataset('train', 0.3)
neighbor_limits = [34, 32, 34, 34, 39]
dloader, _ = get_dataloader(kitti, 0.3, 3.0, 5, 16, True) #, neighbor_limits
loss_fn = HardestContrastiveLoss(0.45, 1.0, pos_margin=0.1, neg_margin=1.4, num_pos=64, num_hn=1024, num_rn=64)
iterator = dloader.__iter__() # dataset iterator


# create optimizer
config = Munch()
config['optimizer'] = 'SGD'
config['momentum'] = 0.98
config['scheduler_gamma'] = 0.1**(1/80)
config['weight_decay'] = 1e-6
config['learning_rate'] = 1e-3
config['iterations'] = 25000
config['resume'] = 0
config['save'] = 500

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
    weights_path = './checkpoints/kitti_HCL64_augm_%d.pth'%config.resume
    model.load_state_dict(torch.load(weights_path))
    print('load previous logs ...')

    with np.load('./logs/kitti_HCL64_augm.npz') as data:
        losses = data['loss'].tolist()
        det_losses = data['det_loss'].tolist()
        desc_losses = data['desc_loss'].tolist()
        pos_desc_losses = data['pos_desc_loss'].tolist()
        neg_desc_losses = data['neg_desc_loss'].tolist()
    for i in range(config.resume // config.save): scheduler.step()
    print(scheduler.get_last_lr())
    print("Done.")


for idx in range(config.resume, config.iterations):
    start = time.time()
    inputs = iterator.next()
    preprocess_time = time.time() - start
    start = time.time()

    pcd0 = inputs['src_pcd']
    pcd1 = inputs['tar_pcd']
    trans = inputs['transform'].numpy()
    for k in pcd0.keys():
        if type(pcd0[k]) == list:
            pcd0[k] = [item.cuda() for item in pcd0[k]]
            pcd1[k] = [item.cuda() for item in pcd1[k]]
        else:
            pcd0[k] = pcd0[k].cuda()
            pcd1[k] = pcd1[k].cuda()
    
    gpu_loading_time = time.time() - start
    start = time.time()
    desc0, saliency0 = model(pcd0)
    inference_time = time.time() - start
    desc1, saliency1 = model(pcd1)

    print('batch %d, preprocessing time: %.4fs, loading time: %.4f, inference time: %.4fs'\
        %(idx+1, preprocess_time, gpu_loading_time, inference_time))    
    print(pcd0['points'][0].shape, pcd1['points'][0].shape)
    print(desc0.shape, desc1.shape, saliency0.shape, saliency1.shape)

    corr = loss_fn.get_matches_indices(pcd0['points'][0].cpu().numpy(), pcd1['points'][0].cpu().numpy(), trans)
    xyz0, xyz1, trans = pcd0['points'][0], pcd1['points'][0], inputs['transform'].cuda()
    xyz1 = xyz1 @ trans[:3, :3].T + trans[:3, -1:].T
    pos_desc_loss, neg_desc_loss, det_loss = loss_fn.loss(corr, xyz0, xyz1, desc0, desc1, saliency0, saliency1)
    loss = pos_desc_loss*2. + neg_desc_loss + det_loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    pos_desc_losses.append(pos_desc_loss.cpu().item())
    neg_desc_losses.append(neg_desc_loss.cpu().item())
    desc_losses.append(pos_desc_losses[-1]*2. + neg_desc_losses[-1])
    det_losses.append(det_loss.cpu().item())
    losses.append(loss.cpu().item())
    print(trans, corr.shape)
    print('loss: %.4f, descriptor loss: %.4f (positive:%.4f negative:%.4f), detection loss: %.4f\n'\
        %(losses[-1], desc_losses[-1], pos_desc_losses[-1], neg_desc_losses[-1], det_losses[-1]))
    
    if idx > 0 and (idx+1) % config.save == 0:
        weights_path = './checkpoints/kitti_HCL64_augm_%d.pth'%(idx+1)
        if idx > 10000: torch.save(model.state_dict(), weights_path)
        scheduler.step()


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


np.savez('./logs/kitti_HCL64_augm.npz',
         loss=np.array(losses),
         det_loss=np.array(det_losses),
         desc_loss=np.array(desc_losses),
         pos_desc_loss=np.array(pos_desc_losses),
         neg_desc_loss=np.array(neg_desc_losses),
        )