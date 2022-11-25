# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions

import torch
from torch.utils.data import DataLoader
import configargparse
from functools import partial

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--resolution', type=int, default=64)
p.add_argument('--batch_size', type=int, default=64)
p.add_argument('--lr', type=float, default=3e-6, help='learning rate. default=5e-5')
p.add_argument('--lr_inner', type=float, default=1e-2, help='learning rate. default=5e-5')
p.add_argument('--num_iters_inner', type=int, default=3, help='learning rate. default=5e-5')

p.add_argument('--num_epochs', type=int, default=51,
               help='Number of epochs to train for.')
p.add_argument('--kl_weight', type=float, default=1e-1,
               help='Weight for l2 loss term on code vectors z (lambda_latent in paper).')
p.add_argument('--fw_weight', type=float, default=1e2,
               help='Weight for the l2 loss term on the weights of the sine network')
p.add_argument('--train_sparsity_range', type=int, nargs='+', default=[200, 400],
               help='Two integers: lowest number of sparse pixels sampled followed by highest number of sparse'
                    'pixels sampled when training the conditional neural process')

p.add_argument('--epochs_til_ckpt', type=int, default=10,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=1000,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--dataset', type=str, default='celeba_64x64',
               help='Time interval in seconds until tensorboard summary is saved.')
# p.add_argument('--dataset', type=str, default='celeba_32x32',
#                help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--model_type', type=str, default='sine',
               help='Nonlinearity for the hypo-network module')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

p.add_argument('--conv_encoder', action='store_true', default=False, help='Use convolutional encoder process')

p.add_argument('--arch_type', type=str, default='latentmodulatedsiren')
p.add_argument('--trainer_type', type=str, default='meta-train')

opt = p.parse_args()


#assert opt.dataset == 'celeba_32x32'
if opt.conv_encoder:
    gmode = 'conv_cnp'
else:
    gmode = 'cnp'

if 'auto-encoder' in opt.arch_type:
    gmode = 'auto-encoder'

#image_resolution = (32, 32)
image_resolution = (opt.resolution, opt.resolution)

#img_dataset = dataio.CelebA(split='train', downsampled=True, resolution=opt.resolution)
img_dataset = dataio.CelebAHQ(split='train', downsampled=True, resolution=opt.resolution)
coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=image_resolution)
generalization_dataset = dataio.ImageGeneralizationWrapper(coord_dataset,
                                                           train_sparsity_range=opt.train_sparsity_range,
                                                           generalization_mode=gmode)


dataloader = DataLoader(generalization_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

if 'latentmodulatedsiren' in opt.arch_type:
    if 'instance' in opt.arch_type:
        model = meta_modules.LatentModulatedSiren(out_channels=img_dataset.img_channels,latent_vector_type='instance')
    else:
        model = meta_modules.LatentModulatedSiren(out_channels=img_dataset.img_channels)
if 'auto-encoder' in opt.arch_type:
    model = meta_modules.AsymmetricAutoEncoder(out_channels=img_dataset.img_channels)
else:
    assert False

model.cuda()

# Define the loss
#loss_fn = partial(loss_functions.image_hypernetwork_loss, None, opt.kl_weight, opt.fw_weight)
#loss_fn = partial(loss_functions.image_mse, None, opt.kl_weight, opt.fw_weight)
loss_fn = partial(loss_functions.image_mse, None)
summary_fn = partial(utils.write_image_summary_small, image_resolution, None)

root_path = os.path.join(opt.logging_root, opt.experiment_name)

if opt.trainer_type == 'meta-train':
    training.meta_train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
                        num_iters_inner=opt.num_iters_inner, lr_inner=opt.lr_inner,
                        steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                        model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, clip_grad=True)
elif opt.trainer_type == 'multi-domain':
    training.multidomain_train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
                               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, clip_grad=True)    
elif opt.trainer_type == 'auto-encoder':
    training.autoencoder_train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
                               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, clip_grad=True)    
