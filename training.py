'''Implements a generic training loop.
'''

import copy
import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import PeakSignalNoiseRatio
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import math
import shutil
import torchvision

import meta_modules, modules, utils, loss_functions


def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None):

    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    #optim = torch.optim.Adam(lr=lr, betas=(0.9, 0.99), params=model.net.parameters())
    #optim = torch.optim.Adam(lr=lr, betas=(0.9, 0.99), params=model.positional_encoding.parameters())

    # copy settings from Raissi et al. (2019) and here 
    # https://github.com/maziarraissi/PINNs
    if use_lbfgs:
        optim = torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
                                  history_size=50, line_search_fn='strong_wolfe')

    if os.path.exists(model_dir):
        val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
        if val == 'y':
            shutil.rmtree(model_dir)

    os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            # if not epoch % epochs_til_checkpoint and epoch:
            #     torch.save(model.state_dict(),
            #                os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
            #     np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
            #                np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()
            
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                if double_precision:
                    model_input = {key: value.double() for key, value in model_input.items()}
                    gt = {key: value.double() for key, value in gt.items()}

                if use_lbfgs:
                    def closure():
                        optim.zero_grad()
                        model_output = model(model_input)
                        losses = loss_fn(model_output, gt)
                        train_loss = 0.
                        for loss_name, loss in losses.items():
                            train_loss += loss.mean() 
                        train_loss.backward()
                        return train_loss
                    optim.step(closure)

                model_output = model(model_input)
                losses = loss_fn(model_output, gt)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                train_losses.append(train_loss.item())
                writer.add_scalar("total_train_loss", train_loss, total_steps)

                # if not total_steps % steps_til_summary:
                #     torch.save(model.state_dict(),
                #                os.path.join(checkpoints_dir, 'model_current.pth'))
                #     summary_fn(model, model_input, gt, model_output, writer, total_steps)

                if not use_lbfgs:
                    optim.zero_grad()
                    train_loss.backward()

                    if clip_grad:
                        if isinstance(clip_grad, bool):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                    optim.step()

                pbar.update(1)

                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

                    if val_dataloader is not None:
                        print("Running validation set...")
                        model.eval()
                        with torch.no_grad():
                            val_losses = []
                            for (model_input, gt) in val_dataloader:
                                model_output = model(model_input)
                                val_loss = loss_fn(model_output, gt)
                                val_losses.append(val_loss)

                            writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                        model.train()

                total_steps += 1

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))


def meta_train(model, train_dataloader, epochs, lr, num_iters_inner, lr_inner,
               steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn, summary_fn,
               val_dataloader=None, double_precision=False, clip_grad=False, loss_schedules=None):

    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    if os.path.exists(model_dir):
        val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
        if val == 'y':
            shutil.rmtree(model_dir)

    os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    keep_params = dict()
    with torch.no_grad():
        for name, param in model.named_parameters():
            keep_params[name] = param.clone()

    # initialise the starting point for the meta gradient (it's faster to copy this than to create new object)
    meta_grad_init = [0 for _ in range(len(model.state_dict()))]

    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        with torch.no_grad():
            for name, param in model.named_parameters():
                param = keep_params[name].clone()

        train_losses = []
        for epoch in range(epochs):
            # if not epoch % epochs_til_checkpoint and epoch:
            #     torch.save(model.state_dict(),
            #                os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
            #     np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
            #                np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()

                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                if double_precision:
                    model_input = {key: value.double() for key, value in model_input.items()}
                    gt = {key: value.double() for key, value in gt.items()}

                # initialise meta-gradient
                meta_grad = copy.deepcopy(meta_grad_init)
                
                # reset context-params
                model.reset_context_params(model_input['coords'].size(0))

                for _ in range(num_iters_inner):                    
                    pred_inner = model(model_input)
                    loss_inner = loss_fn(pred_inner, gt)
                    grad_inner = torch.autograd.grad(loss_inner['img_loss'],
                                                     model.context_params,
                                                     create_graph=True)[0]
                    model.context_params = torch.nn.Parameter(model.context_params.detach() - lr_inner * grad_inner.detach())
                
                model_output = model(model_input)
                losses = loss_fn(model_output, gt)

                # compute gradient for shared parameters                
                task_grad = torch.autograd.grad(losses['img_loss'], model.parameters())

                # add to meta-gradient
                for g in range(len(task_grad)):
                    meta_grad[g] += task_grad[g].detach()

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                train_losses.append(train_loss.item())
                writer.add_scalar("total_train_loss", train_loss, total_steps)

                # if not total_steps % steps_til_summary:
                #     torch.save(model.state_dict(),
                #                os.path.join(checkpoints_dir, 'model_current.pth'))
                #     summary_fn(model, model_input, gt, model_output, writer, total_steps)

                optim.zero_grad()

                # set gradients of parameters manually
                for c, param in enumerate(model.parameters()):
                    param.grad = meta_grad[c]

                    if clip_grad:
                        if isinstance(clip_grad, bool):
                            torch.nn.utils.clip_grad_norm_(param, max_norm=10.)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                optim.step()

                pbar.update(1)

                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

                    if val_dataloader is not None:
                        print("Running validation set...")
                        model.eval()
                        with torch.no_grad():
                            val_losses = []
                            for (model_input, gt) in val_dataloader:
                                model_output = model(model_input)
                                val_loss = loss_fn(model_output, gt)
                                val_losses.append(val_loss)

                            writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                        model.train()

                total_steps += 1

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        # np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
        #            np.array(train_losses))


def multidomain_train(model, train_dataloader, epochs, lr,
                      steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn, summary_fn,
                      val_dataloader=None, double_precision=False, clip_grad=False, loss_schedules=None):

    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    if os.path.exists(model_dir):
        val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
        if val == 'y':
            shutil.rmtree(model_dir)

    os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    keep_params = dict()
    with torch.no_grad():
        for name, param in model.named_parameters():
            keep_params[name] = param.clone()

    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        with torch.no_grad():
            for name, param in model.named_parameters():
                param = keep_params[name].clone()

        train_losses = []
        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()

                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                if double_precision:
                    model_input = {key: value.double() for key, value in model_input.items()}
                    gt = {key: value.double() for key, value in gt.items()}
               
                # reset context-params
                model.reset_context_params(model_input['coords'].size(0))
                model_output = model(model_input)
                losses = loss_fn(model_output, gt)
   
                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                train_losses.append(train_loss.item())
                writer.add_scalar("total_train_loss", train_loss, total_steps)

                if not total_steps % steps_til_summary:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))
                    #summary_fn(model, model_input, gt, model_output, writer, total_steps)

                optim.zero_grad()
                losses['img_loss'].backward()
                optim.step()

                pbar.update(1)

                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

                    if val_dataloader is not None:
                        print("Running validation set...")
                        model.eval()
                        with torch.no_grad():
                            val_losses = []
                            for (model_input, gt) in val_dataloader:
                                model_output = model(model_input)
                                val_loss = loss_fn(model_output, gt)
                                val_losses.append(val_loss)

                            writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                        model.train()

                total_steps += 1

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))


def autoencoder_train(model, train_dataloader, epochs, lr,
                      steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn, summary_fn,
                      val_dataloader=None, double_precision=False, clip_grad=False, loss_schedules=None):
    
    optim = torch.optim.Adam(lr=lr, params=model.get_parameters())

    if os.path.exists(model_dir):
        val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
        if val == 'y':
            shutil.rmtree(model_dir)

    os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    sample_dir = os.path.join(model_dir, 'samples')
    utils.cond_mkdir(sample_dir)

    writer = SummaryWriter(summaries_dir)
    
    total_steps = 0
    keep_params = dict()
    with torch.no_grad():
        for name, param in model.named_parameters():
            keep_params[name] = param.clone()

    PSNR = PeakSignalNoiseRatio()

    model.train()
    model.conv_encoder.eval()

    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        with torch.no_grad():
            for name, param in model.named_parameters():
                param = keep_params[name].clone()

        train_losses = []
        train_psnrs = []
        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()

                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                if double_precision:
                    model_input = {key: value.double() for key, value in model_input.items()}
                    gt = {key: value.double() for key, value in gt.items()}
               
                # reset context-params
                model_output = model(model_input)
                losses = loss_fn(model_output, gt)
                
                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss


                train_losses.append(train_loss.item())

                writer.add_scalar("total_train_loss", train_loss, total_steps)

                train_psnr = 0.
                for bi in range(0, len(gt['img'])):
                    train_psnr += PSNR(model_output['model_out'][bi].detach().cpu(), gt['img'][bi].detach().cpu())
                train_psnr = train_psnr/len(gt['img'])
                train_psnrs.append(train_psnr)

                if not total_steps % steps_til_summary:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))
                    #summary_fn(model, model_input, gt, model_output, writer, total_steps)

                optim.zero_grad()
                losses['img_loss'].backward()
                optim.step()

                pbar.update(1)

                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss %0.6f, Total PSNR %2.2f, iteration time %0.6f" % (epoch, train_loss, train_psnr, time.time() - start_time))
                    
                    H = int(math.sqrt(gt['img'].size(1)))
                    gt_grid = torchvision.utils.make_grid(gt['img'].detach().cpu().view(-1, H, H, 3).permute(0, 3, 1, 2))
                    pred_grid = torchvision.utils.make_grid(model_output['model_out'].detach().cpu().view(-1, H, H, 3).permute(0, 3, 1, 2))

                    torchvision.utils.save_image(gt_grid, os.path.join(sample_dir, f'gt_grid_iter{total_steps}.png'))
                    torchvision.utils.save_image(pred_grid, os.path.join(sample_dir, f'pred_grid_iter{total_steps}.png'))

                    if val_dataloader is not None:
                        print("Running validation set...")
                        model.eval()
                        with torch.no_grad():
                            val_losses = []
                            for (model_input, gt) in val_dataloader:
                                model_output = model(model_input)
                                val_loss = loss_fn(model_output, gt)
                                val_losses.append(val_loss)

                            writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                        model.train()
                        model.conv_encoder.eval()

                total_steps += 1

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))
        np.savetxt(os.path.join(checkpoints_dir, 'train_psnrs_final.txt'),
                   np.array(train_psnrs))


def memorize(model, batch_data, epochs, lr, loss_fn, clip_grad=False):

    PSNR = PeakSignalNoiseRatio()
    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    total_steps = 0
    

    for epoch in range(epochs):
        model_input, gt = batch_data
        
        model_input = {key: value.cuda() for key, value in model_input.items()}
        gt = {key: value.cuda() for key, value in gt.items()}

        model_output = model(model_input)
        losses = loss_fn(model_output, gt)

        train_loss = 0.
        for loss_name, loss in losses.items():
            single_loss = loss.mean()
            train_loss += single_loss      

        optim.zero_grad()
        train_loss.backward()

        if clip_grad:
            if isinstance(clip_grad, bool):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

        optim.step()

    psnr = PSNR(model_output['model_out'][0].cpu() * 0.5 + 0.5, gt['img'][0].cpu() * 0.5 + 0.5)
    
    return [float(train_loss), float(psnr)], model_output


class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)
