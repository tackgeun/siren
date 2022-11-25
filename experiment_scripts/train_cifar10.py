# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions, modules

from torch.utils.data import DataLoader
import configargparse
from functools import partial
from torchvision.datasets import CIFAR10
from torch.nn import functional as F
from torchvision import datasets, transforms
import pdb

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--dataset', type=str, default='cifar10')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')


# General training options
p.add_argument('--batch_size', type=int, default=128)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--num_epochs', type=int, default=10,
               help='Number of epochs to train for.')
p.add_argument('--num-iters-inner', type=int, default=3)
p.add_argument('--lr-inner', type=float, default=3)

p.add_argument('--epochs_til_ckpt', type=int, default=25,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=1000,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                    '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu),'
                    'and in the future: "mixed" (first layer sine, other layers tanh)')

p.add_argument('--modulation_type', type=str, default='none')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
opt = p.parse_args()

# transform = transforms.Compose([transforms.Resize((32, 32)),
#                                   transforms.ToTensor(),
#                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                                  ])

if opt.dataset == 'cifar10':
    img_dataset = CIFAR10(root='datasets/cifar10', train=True, download=True)
    RES = 32

coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=RES, compute_diff=None, istuple=True)
image_resolution = (RES, RES)

dataloader = DataLoader(coord_dataset, shuffle=False, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

# Define the model.
if opt.model_type == 'sine' or opt.model_type == 'relu' or opt.model_type == 'tanh' or opt.model_type == 'selu' or opt.model_type == 'elu'\
        or opt.model_type == 'softplus':
    model = modules.MetaBVPNet(out_features=3, type=opt.model_type, mode='mlp', sidelength=image_resolution, modulation_type=opt.modulation_type)
else:
    raise NotImplementedError
model.cuda()

root_path = os.path.join(opt.logging_root, opt.experiment_name)

# Define the loss
loss_fn = partial(loss_functions.image_mse, None)
summary_fn = partial(utils.write_image_summary, image_resolution, compute_diff=None)

training.meta_training(model=model, train_dataloader=dataloader,
                       epochs=opt.num_epochs, lr=opt.lr,
                       num_iters_inner=opt.num_iters_inner, lr_inner=opt.lr_inner,
                       steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                       model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn)

