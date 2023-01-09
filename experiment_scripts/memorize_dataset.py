# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import shutil

from tqdm.autonotebook import tqdm
import configargparse
from functools import partial

import torch
from torch.utils.data import DataLoader
import torchvision
import numpy as np

import dataio, training, modules, loss_functions, utils

import pdb

def get_model(opt):
    # define the model.
    if opt.model_type == 'sine' or opt.model_type == 'relu' or opt.model_type == 'tanh' or opt.model_type == 'selu' or opt.model_type == 'elu'\
            or opt.model_type == 'softplus':
        model = modules.SingleBVPNet(type=opt.model_type, mode='mlp', sidelength=image_resolution,
                                    sparse_matrix=opt.sparse_type,
                                    out_features=3,
                                    hidden_features=opt.hidden_features,
                                    num_hidden_layers=opt.num_hidden_layers)
    elif opt.model_type == 'rbf' or opt.model_type == 'nerf' or opt.model_type == 'instant-ngp':
        model = modules.SingleBVPNet(type='relu', mode=opt.model_type, sidelength=image_resolution,
                                    out_features=3,
                                    sparse_matrix=opt.sparse_type,
                                    hidden_features=opt.hidden_features,
                                    num_hidden_layers=opt.num_hidden_layers)
    else:
        raise NotImplementedError

    return model

p = configargparse.ArgumentParser()

# Experiment
p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
p.add_argument('--save_images', action='store_true')

# Dataset
p.add_argument('--dataset', type=str, default='celeba',
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--resolution', type=int, default=64)
               
# Optimization
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--lr', type=float, default=1e-3, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=1000,
               help='Number of epochs to train for.')

# Model
p.add_argument('--model_type', type=str, default='sine',
               help='Nonlinearity for the hypo-network module')
p.add_argument('--sparse_type', type=str, default='none')
p.add_argument('--hidden_features', type=int, default=32)
p.add_argument('--num_hidden_layers', type=int, default=3)
opt = p.parse_args()

# Dataset
image_resolution = (opt.resolution, opt.resolution)

if 'celeba' in opt.dataset.lower():
    img_dataset = dataio.CelebAHQ(split='train', downsampled=True, resolution=opt.resolution)
elif 'ffhq' in opt.dataset.lower():
    #img_dataset = dataio.FFHQsample(opt.resolution)
    assert opt.resolution == 256
    img_dataset = dataio.FFHQ(split='train')
# coordinate dataset
coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=opt.resolution, compute_diff='none')
image_resolution = (opt.resolution, opt.resolution)

dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

# Checkpoint
model_dir = os.path.join(opt.logging_root, opt.experiment_name)

# if os.path.exists(model_dir):
#     val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
#     if val == 'y':
#         shutil.rmtree(model_dir)

utils.cond_mkdir(model_dir)

checkpoints_dir = os.path.join(model_dir, 'checkpoints')
utils.cond_mkdir(checkpoints_dir)

losses_dir = os.path.join(model_dir, 'losses')
utils.cond_mkdir(losses_dir)

images_dir = os.path.join(model_dir, 'images')
utils.cond_mkdir(images_dir)

# Define the loss
loss_fn = partial(loss_functions.image_mse, None)


model = get_model(opt)
print(model)
# count parameters
nparameters = 0
nparams = []
for param in model.parameters():
    nparam = param.nelement()
    nparams.append(nparam)
    nparameters += nparam
print("Parameter count: ", nparameters)    
for nparam in nparams:
    print(nparam)

idx = 0
# Memorizing Loop
with tqdm(total=len(dataloader)) as pbar:
    for sample_idx, batch_data in enumerate(dataloader):
        idx = int(batch_data[0]['idx'])
        if not os.path.exists(os.path.join(checkpoints_dir, f'model{idx}.pth')):
            model = get_model(opt)

            model.cuda()
            # memorizing loop
            train_losses, outputs = training.memorize(model, batch_data, opt.num_epochs, lr=opt.lr, loss_fn=loss_fn)
            
            torch.save(model.state_dict(),
                    os.path.join(checkpoints_dir, f'model{idx}.pth'))
            np.savetxt(os.path.join(losses_dir, f'train_losses{idx}.txt'),
                    np.array(train_losses))

            if opt.save_images:
                # img = img.permute(1, 2, 0).view(-1, self.img_channels)
                img = outputs['model_out'][0].cpu().view(opt.resolution, opt.resolution, 3)
                img = img.permute(2, 0, 1) * 0.5 + 0.5
                torchvision.utils.save_image(img, os.path.join(images_dir, f'recon{idx}.png'))
                # img = batch_data[1]['model_out'][0].cpu().view(3, opt.resolution, opt.resolution)
                # torchvision.utils.save_image(img, os.path.join(images_dir, f'gt{idx}.png'))

            pbar.set_description(f'mse_loss:{train_losses[0]:.4f} PSNR:{train_losses[1]:.2f}')

        pbar.update(1)
