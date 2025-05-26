import os
import utils
import torch
import torch.nn as nn
from torchvision import transforms, utils as tv_utils
from torch.utils.data import DataLoader
import numpy as np
import data
import scipy.io as sio
from options.training_options import TrainOptions
import time
from models import AutoEncoderCov3DMem, EntropyLossEncap
import matplotlib.pyplot as plt
from PIL import Image

# Parse options
opt_parser = TrainOptions()
opt = opt_parser.parse(is_print=True)
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

# Settings
opt.NumWorker = 4
opt.ModelName = 'MemAE'
opt.ModelSetting = 'Conv3DSpar'
opt.Dataset = 'data_lin_256'
opt.ImgChnNum = 3

# Reproducibility
torch.manual_seed(opt.Seed)
if opt.IsDeter:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Model parameters
batch_size = opt.BatchSize
lr = opt.LR
epochs = opt.EpochNum
mem_dim = opt.MemDim
ent_weight = opt.EntropyLossWeight
shrink_thres = opt.ShrinkThres

# Data paths
data_root = os.path.join(opt.DataRoot, opt.Dataset)
train_frames = os.path.join(data_root, 'Train')
train_idx = os.path.join(data_root, 'Train_idx')

# Output dirs
model_dir = os.path.join(opt.ModelRoot, f"model_{opt.ModelSetting}")
vis_dir = os.path.join(model_dir, 'visualizations')
os.makedirs(vis_dir, exist_ok=True)

# Data loader setup
norm_mean = (0.5,) * opt.ImgChnNum
norm_std = (0.5,) * opt.ImgChnNum
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])

dataset = data.VideoDatasetPy(train_idx, train_frames, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=opt.NumWorker)

# Model
model = AutoEncoderCov3DMem(opt.ImgChnNum, mem_dim, shrink_thres=shrink_thres)
model.apply(utils.weights_init)
model.to(device)

# Losses and optimizer
recon_criterion = nn.MSELoss().to(device)
entropy_criterion = EntropyLossEncap().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Tracking lists
loss_list, recon_list, ent_list = [], [], []
iteration = 0

# Training loop
for epoch in range(epochs):
    for batch_idx, (_, frames) in enumerate(loader):
        frames = frames.to(device)
        output = model(frames)
        recon = output['output']
        att = output['att']

        recon_loss = recon_criterion(recon, frames)
        ent_loss = entropy_criterion(att)
        loss = recon_loss + ent_weight * ent_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record losses
        loss_list.append(loss.item())
        recon_list.append(recon_loss.item())
        ent_list.append(ent_loss.item())

        # Save snapshots
        if iteration % opt.SnapInterval == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, f"{opt.ModelSetting}_snap.pt"))

        # Save visualizations periodically
        if iteration % opt.TBImgLogInterval == 0:
            # frames vis
            vis = utils.vframes2imgs(utils.UnNormalize(mean=norm_mean, std=norm_std)(frames).cpu(), step=5, batch_idx=0)
            recon_vis = utils.vframes2imgs(utils.UnNormalize(mean=norm_mean, std=norm_std)(recon).cpu(), step=5, batch_idx=0)
            # concatenate and save
            concat = np.concatenate(vis + recon_vis, axis=1)  # side by side
            img = Image.fromarray((concat * 255).astype(np.uint8))
            img.save(os.path.join(vis_dir, f"vis_iter_{iteration}.png"))

        iteration += 1

    # Save model at end of epoch
    if (epoch % opt.SaveCheckInterval) == 0:
        torch.save(model.state_dict(), os.path.join(model_dir, f"{opt.ModelSetting}_epoch_{epoch:04d}.pt"))

# Final save
torch.save(model.state_dict(), os.path.join(model_dir, f"{opt.ModelSetting}_final.pt"))

# Plot and save loss curves
plt.figure()
plt.plot(loss_list, label='Total Loss')
plt.plot(recon_list, label='Recon Loss')
plt.plot(ent_list, label='Entropy Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss Curves')
plt.savefig(os.path.join(vis_dir, 'loss_curve.png'))
plt.close()
