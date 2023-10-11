#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import os
import sys
from tqdm import tqdm
from pathlib import Path

import numpy as np
import nibabel as nib   
import pandas as pd 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utilities.adversarial_loss import GANLoss
from utilities.dataset import DatasetRDO
from utilities.utils import (
    create_logger, 
    poly_lr,
    infinite_iterable,
    init_training_variables)

from network.mhvae import MHVAE2D
from network.discriminator_pathgan2D import NLayerDiscriminator
from monai.utils import set_determinism


PHASES = ['training', 'validation']

# Number of worker
workers = 10

# Moving average validation
val_eval_criterion_alpha = 0.95

def save(pred, affine, path):
    pred = (255*(pred+1)/2).int()
    pred = pred.permute(2,3,0,1).squeeze().cpu().numpy()
    nib.Nifti1Image(pred, affine).to_filename(path)

def train(paths_dict, model, netD, device, logger, opt):
    
    since = time.time()

    logger.info(f"Modalities are {opt.modalities}")
    # Define transforms for data normalization and augmentation
    dataloaders = dict()
    subjects_dataset = dict()
    ind_batch = dict()

    for phase in PHASES:
        paths_dict[phase] = {mod:paths_dict[phase][mod] for mod in opt.modalities}
        ind_batch[phase] = np.arange(0, len(paths_dict[phase][opt.modalities[0]]),  opt.batch_size)
        subjects_dataset[phase] = DatasetRDO(paths_dict[phase], mode=phase)
        dataloaders[phase] = infinite_iterable(DataLoader(subjects_dataset[phase], batch_size=opt.batch_size, shuffle=phase=='training', num_workers=workers))

    # Training parameters 
    df_path = os.path.join(opt.model_dir,'log.csv')
    save_path = os.path.join(opt.model_dir, 'models', './CP_{}_{}.pth')
    df, val_eval_criterion_MA, best_val_eval_criterion_MA, best_epoch, epoch, initial_lr = init_training_variables(model, netD, opt, logger)

    # Optimisation policy main model
    optimizer = torch.optim.Adamax(model.parameters(), lr=initial_lr, betas=(0.9, 0.999,))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, opt.epochs - epoch, eta_min=1e-4)

    # Optimisation policy discriminators
    lr0_disct = 2e-4
    d_optimizer = torch.optim.Adam([{'params': netD[mod].parameters()} for mod in opt.modalities], lr=lr0_disct)

    gan = GANLoss('lsgan').to(device)
    first_mod = opt.modalities[0]

    continue_training = True
    
    assert opt.batch_size ==1, 'Only batch_size==1 supported'

    while continue_training:
        epoch+=1
        logger.info('-' * 10)
        logger.info('Epoch {}/'.format(epoch))

        for param_group in optimizer.param_groups:
            logger.info("Current learning rate for main model is: {}".format(param_group['lr']))

        for param_group in d_optimizer.param_groups:
            logger.info("Current learning rate for discriminators is: {}".format(param_group['lr']))

        if opt.warmup_value_null>epoch:
            w_kl = 0.0
            w_dis = 0.0
        else:
            w_kl = min(opt.w_kl, opt.w_kl*(epoch - opt.warmup_value_null) / (opt.warmup_value - opt.warmup_value_null))
            if epoch<opt.warmup_value_null_disc:
                w_dis = 0.0
            else:
                w_dis = opt.w_dis

        logger.info("Current w_kl {}".format(w_kl))
        logger.info("Current w_dis {}".format(w_dis))
        

        # Each epoch has a training and validation phase
        for phase in ['training', 'validation']:
            logger.info(phase)
            if phase == 'training':
                model.train()  # Set model to training mode
                for mod in opt.modalities:
                    netD[mod].train()
            else:
                model.eval()  # Set model to evaluate mode
                for mod in opt.modalities:
                    netD[mod].eval()

            running_loss = 0.0
            running_loss_img = 0.0
            running_kl = 0.0
            running_kl_multi = []
            running_loss_adv = 0.0
            running_loss_disc = 0.0
            epoch_samples = 0

            # Iterate over data
            for id_b in tqdm(ind_batch[phase]):
                batch = next(dataloaders[phase])
                name = batch[f"{first_mod}_name"][0]
                if phase=='training':
                    nw = 16
                    w = np.random.randint(0, batch[first_mod].shape[-1]-nw-1)
                else:
                    w = 0
                    nw = batch[mod].shape[-1]
                imgs = dict()
                for mod in opt.modalities:
                    imgs[mod] = batch[mod].permute(0,4,1,2,3).reshape(-1, 1, *batch[mod].shape[2:4])[w:w+nw:,...]
                    imgs[mod] = imgs[mod].to(device).float() / 255.
                    imgs[mod] =  2*imgs[mod] - 1 
                    
                nb_voxels = np.prod(imgs[first_mod].shape) / len(opt.modalities)
                mask = 2*(imgs[first_mod]>-1) - 1
                
                if epoch==1 and id_b==0 and phase=='training':
                    with torch.set_grad_enabled(False):
                        logger.info('[INFO] Display infos latent')
                        model({mod:imgs[first_mod]}, verbose=True)

                with torch.set_grad_enabled(phase == 'training'):
                    optimizer.zero_grad()
                    g_loss = 0.0
                    loss_img = 0.0
                    kl = 0.0
                    kl_multi = 0.0
                    
                    # Compute for for all modalities
                    output_img, kls  = model({mod:imgs[mod].clone() for mod in opt.modalities})
                    output_img = 2*((output_img+1)/2*(mask>=0).float()) - 1
                    kl += torch.sum(torch.stack(kls['prior']))
                    # kl_multi +=  sum([torch.sum(torch.stack(kls[mod_kl])) for mod_kl in opt.modalities]) 
                    
                    loss_img += nn.SmoothL1Loss(reduction='sum')(output_img, torch.cat([imgs[mod] for mod in opt.modalities], 1))
                    
                    # Compute for for each modality independently (OK iif N=2)
                    for mod in opt.modalities:
                        output_img, kls  = model({mod:imgs[mod]})
                        kl += torch.sum(torch.stack(kls['prior']))
                        loss_img += nn.SmoothL1Loss(reduction='sum')(output_img, torch.cat([imgs[mod] for mod in opt.modalities], 1))  
                        if w_dis>0:
                            for j, mod_adv in enumerate(opt.modalities ):
                                other =  output_img[:,j:j+1,...]
                                g_loss += nb_voxels*gan(netD[mod_adv](other), True)     
                                                           
                    loss = loss_img + w_kl*kl + w_dis*g_loss

                    # backward + optimize only if in training phase
                    if phase == 'training':
                        if loss>0:
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                
                    # Train Discriminator
                    if phase == 'training' and epoch>opt.warmup_value_disc:
                        d_optimizer.zero_grad()
                        with torch.set_grad_enabled(phase == 'training'):
                            d_loss = 0.0
                            for i, mod in enumerate(opt.modalities):
                                input_mod = imgs[mod].to(device)
                                d_loss += gan(netD[mod](input_mod.contiguous().detach()), True)
                                other =  output_img[:,i:i+1,...].detach()
                                d_loss += gan(netD[mod](other), False) 

                            optimizer.zero_grad() 
                            d_loss.backward()
                            d_optimizer.step()
                    else:
                        d_loss = 0.0
                                
                # statistics
                epoch_samples += 1
                running_loss += loss
                running_loss_img += loss_img
                running_loss_adv += g_loss
                running_kl += kl
                if kl_multi>0:
                    running_kl_multi.append(kl_multi)
                running_loss_disc += d_loss

            running_loss = running_loss.item()
            if phase=='training':
                lr_scheduler.step()

            logger.info('{}  Loss Total: {:.4f}'.format(
                phase, running_loss / epoch_samples))

            logger.info('{}  Loss Img: {:.4f}'.format(
                phase, running_loss_img / epoch_samples))

            logger.info('{}  Loss KL: {:.4f}'.format(
                phase, running_kl / epoch_samples))
            
            logger.info('{}  Loss KL multi: {:.4f}'.format(
                phase, sum(running_kl_multi)/max(1,len(running_kl_multi))))

            logger.info('{}  Loss Adversarial: {:.4f}'.format(
                phase, running_loss_adv / epoch_samples))

            logger.info('{}  Loss Discriminator: {:.4f}'.format(
                phase, running_loss_disc / epoch_samples))

            epoch_loss = running_loss / epoch_samples

            if phase=='validation':
                if val_eval_criterion_MA is None: # first iteration
                    val_eval_criterion_MA = epoch_loss
                    best_val_eval_criterion_MA = val_eval_criterion_MA

                else: #update criterion
                    val_eval_criterion_MA = val_eval_criterion_alpha * val_eval_criterion_MA + (
                                1 - val_eval_criterion_alpha) * epoch_loss

                if val_eval_criterion_MA < best_val_eval_criterion_MA:
                    best_val_eval_criterion_MA = val_eval_criterion_MA
                    best_epoch = epoch
                    torch.save(model.state_dict(), save_path.format('main','best'))
            
                df = df.append({'epoch':epoch,
                    'best_epoch':best_epoch,
                    'MA':val_eval_criterion_MA,
                    'best_MA':best_val_eval_criterion_MA,  
                    'lr':param_group['lr']}, ignore_index=True)
                df.to_csv(df_path, index=False)

                if epoch==opt.epochs:
                    continue_training=False
                    torch.save(model.state_dict(), save_path.format('main', 'final'))

        if epoch%10==0 or epoch<10:
            with torch.no_grad():
                
                torch.save(model.state_dict(), save_path.format('main', epoch))
                for mod in opt.modalities:
                    torch.save(netD[mod].state_dict(), save_path.format(mod, epoch))
                    
                affine = batch[f"{first_mod}_affine"][0].cpu().numpy().squeeze()
                name = batch[f"{first_mod}_name"]
                saving_path = os.path.join(opt.model_dir, 'samples')
                
                pred, _  = model({mod:imgs[mod] for mod in opt.modalities})
                pred = 2*((pred+1)/2*(mask>=0).float()) - 1
                name = batch[f"{first_mod}_name"][0]+'_'+str(epoch)+'_full.nii.gz'
                save(pred, affine, os.path.join(saving_path,name))
                
                for mod in opt.modalities:
                    pred, _  = model({mod:imgs[mod]})
                    pred = 2*((pred+1)/2*(mask>=0).float()) - 1
                    name = batch[f"{mod}_name"][0]+'_'+str(epoch)+'_single.nii.gz'
                    save(pred, affine, os.path.join(saving_path,name))
                
                pred = model.sample(batch_size=mask.size(0))
                pred = 2*((pred+1)/2*(mask>=0).float()) - 1
                name = batch[f"{first_mod}_name"][0]+'_'+str(epoch)+'_sample.nii.gz'
                save(pred, affine, os.path.join(saving_path,name))

    time_elapsed = time.time() - since
    logger.info('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info('Best epoch is {}'.format(best_epoch))

def main():
    opt = parsing_data()

    if opt.warmup_value<0:
        opt.warmup_value = opt.epochs

    # FOLDERS
    fold_dir = opt.model_dir
    fold_dir_model = os.path.join(fold_dir,'models')
    if not os.path.exists(fold_dir_model):
        os.makedirs(fold_dir_model)
    
    output_path = os.path.join(fold_dir,'samples')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logger = create_logger(fold_dir)
    logger.info("[INFO] Hyperparameters")
    logger.info(f"Batch size: {opt.batch_size}")
    logger.info(f"Initial lr: {opt.ini_lr}")
    logger.info(f"Total number of epochs: {opt.epochs}")
    logger.info(f"Number of modalities: {len(opt.modalities)}")
    logger.info(f"With residual: {opt.no_res}")
    logger.info(f"With se: {opt.no_se}")
    
    set_determinism(seed=opt.seed)

    if torch.cuda.is_available():
        logger.info('[INFO] GPU available.')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        raise Exception(
            "[INFO] No GPU found or Wrong gpu id, please run without --cuda")
        
        
    logger.info("[INFO] Reading data")
    # PHASES
    split_path = os.path.join(opt.dataset_split)
    df_split = pd.read_csv(split_path,header =None)
    list_file = dict()
    for phase in PHASES: # list of patient name associated to each phase
        list_file[phase] = df_split[df_split[1].isin([phase])][0].tolist()

    paths_dict = {phase:{modality:[] for modality in opt.modalities} for phase in PHASES}
    for phase in PHASES:
        for subject in list_file[phase]:
            if os.path.exists(opt.path_data+subject+opt.modalities[0]+'.nii.gz'):
                for modality in opt.modalities:
                    paths_dict[phase][modality].append(opt.path_data+subject+modality+'.nii.gz')

    # MODEL 
    logger.info("[INFO] Building model")  
    output_feat = len(opt.modalities)
    
    model = MHVAE2D(
        modalities=opt.modalities,   
        base_num_features=opt.base_features,   
        num_pool=opt.pools,   
        num_img=output_feat,
        original_shape=opt.spatial_shape[:2],
        max_features=opt.max_features,
        with_residual=opt.no_res,
        with_se=opt.no_se,
        logger=logger,
        ).to(device)
        
    discriminators = {mod: NLayerDiscriminator(input_nc=1, n_class=1, ndf=64, n_layers=6).to(device) for mod in opt.modalities}

    logger.info("[INFO] Training")
    
    train(paths_dict, 
        model, 
        discriminators, 
        device, 
        logger,
        opt)



def parsing_data():
    parser = argparse.ArgumentParser(
        description='Training ')


    parser.add_argument('--model_dir',
                        type=str)

    parser.add_argument('--dataset_split',
                        type=str,
                        default='/workspace/home/MVAE/splits/split.csv')


    parser.add_argument('--path_data',
                        type=str,
                        default='../data/golby_data/iUS-T2-shape-reg-005mm-crop/')
    
    parser.add_argument('--ini_lr',
                    type=float,
                    default=2e-3)

    parser.add_argument('--w_kl',
                    type=float,
                    default=1.0)
    

    parser.add_argument('--w_dis',
                    type=float,
                    default=1.0)

    parser.add_argument('--seed',
                    type=int,
                    default=2)

    parser.add_argument('--batch_size',
                    type=int,
                    default=6)

    parser.add_argument('--max_features',
                    type=int,
                    default=128)

    parser.add_argument('--base_features',
                    type=int,
                    default=16)

    parser.add_argument('--pools',
                    type=int,
                    default=6)

    parser.add_argument('--epochs',
                    type=int,
                    default=500)

    parser.add_argument('--fold',
                    type=int,
                    default=0)

    parser.add_argument('--warmup_value',
                    type=int,
                    default=-1)

    parser.add_argument('--warmup_value_null',
                    type=int,
                    default=0)
    
    parser.add_argument('--warmup_value_null_disc',
                    type=int,
                    default=100)

    parser.add_argument('--warmup_value_disc',
                    type=int,
                    default=100)

    parser.add_argument('--spatial_shape',
                    type=int,
                    nargs="+",
                    default=(208,208,80))

    parser.add_argument('--modalities',
                    type=str,
                    nargs="+",
                    default=['US', 'T2'])
    
    parser.add_argument('--no_se', action='store_false', help='Squeeze and Excitation disabled')

    parser.add_argument('--no_res', action='store_false', help='Residual connection disabled')
    
    parser.add_argument('--new_res', action='store_true', help='res_net')

    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    main()


