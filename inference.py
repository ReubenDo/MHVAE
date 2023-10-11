#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import os
import sys
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path

from copy import deepcopy
import numpy as np 
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
    init_training_variables,
    save,
    save_feature)

from network.mhvae import MHVAE2D
from network.discriminator_pathgan2D import NLayerDiscriminator


from monai.utils import set_determinism


# Define training and patches sampling parameters   
SPATIAL_SHAPE = (208,208,80)

NB_CLASSES = 3

# Number of worker
workers = 10

# Training parameters
val_eval_criterion_alpha = 0.95
train_loss_MA_alpha = 0.95
nb_patience = 10
patience_lr = 5
weight_decay = 1e-5

MODALITIES = ['0002']
PHASES = ['validation', 'inference']

def run_inference(paths_dict, saving_path, model, device, logger, opt):
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
    save_path = os.path.join(opt.model_dir, 'models', './CP_{}_{}.pth')
    model.load_state_dict(torch.load(save_path.format('main',opt.epochs)))
    logger.info(f"Loading model from {save_path.format('main',opt.epochs)}")
    first_mod = opt.modalities[0]
    
    assert opt.batch_size ==1, 'Only batch_size==1 supported'
    model.eval()  # Set model to evaluate mode

    for phase in PHASES:
        # Iterate over data
        for _ in tqdm(ind_batch[phase]):
            batch = next(dataloaders[phase])
            name = batch[f"{first_mod}_name"][0]
            imgs = dict()
            for mod in opt.modalities:
                imgs[mod] = batch[mod].permute(0,4,1,2,3).reshape(-1, 1, *batch[mod].shape[2:4])
                imgs[mod] = imgs[mod].to(device).float() / 255.
                imgs[mod] =  2*imgs[mod] - 1 
            mask = 2*(imgs[first_mod]>-1) - 1

            with torch.no_grad():      
                affine = batch[f"{first_mod}_affine"][0].cpu().numpy().squeeze()
                name = batch[f"{first_mod}_name"]
                
                name = batch[f"{first_mod}_name"][0].replace(first_mod,'')+'{}_{}.nii.gz'
                
                features = []
                paths_features = []    
                  
                pred, _, feat  = model({mod:imgs[mod] for mod in opt.modalities}, return_feat=True, temp=opt.temp)
                if opt.save_images:
                    pred = 2*((pred+1)/2*(mask>=0).float()) - 1
                    save(pred, affine, os.path.join(saving_path, name.format('full','img')))
                
                if opt.save_features:
                    features.append(feat)
                    paths_features.append(os.path.join(saving_path, name.format('full', 'feat')))
                
                
                for mod in opt.modalities:
                    pred, _, feat  = model({mod:imgs[mod]}, return_feat=True, temp=opt.temp)
                    if opt.save_images:
                        pred = 2*((pred+1)/2*(mask>=0).float()) - 1
                        save(pred, affine, os.path.join(saving_path,name.format(mod, 'img')))
                    if opt.save_features:
                        features.append(feat)
                        paths_features.append(os.path.join(saving_path,name.format(mod, 'feat')))
                if opt.save_features:    
                    save_feature(features, paths_features, affine)     
                
                if opt.save_samples:
                    pred = model.sample(mask.size(0))
                    pred = 2*((pred+1)/2*(mask>=0).float()) - 1                    
                    save(pred, affine, os.path.join(saving_path,name.format('prior',opt.temp)))


def main():
    opt = parsing_data()

    # FOLDERS
    fold_dir = opt.model_dir
    
    output_path = os.path.join(fold_dir,f'inference_{opt.temp}')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    output_path_logs = os.path.join(fold_dir,'logs')
    if not os.path.exists(output_path_logs):
        os.makedirs(output_path_logs)

    logger = create_logger(output_path_logs)
    logger.info("[INFO] Hyperparameters")
    logger.info(f"Batch size: {opt.batch_size}")
    logger.info(f"Number of modalities: {len(opt.modalities)}")
    logger.info(f"With residual: {opt.no_res}")
    logger.info(f"With se: {opt.no_se}")
    logger.info(f"Temperature: {opt.temp}")
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
            subject_data = dict()
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

    logger.info("[INFO] Running inference")
    
    run_inference(
        paths_dict, 
        output_path,
        model, 
        device, 
        logger,
        opt)



def parsing_data():
    parser = argparse.ArgumentParser(
        description='3D Segmentation Using PyTorch and NiftyNet')


    parser.add_argument('--model_dir',
                        type=str)

    parser.add_argument('--dataset_split',
                        type=str,
                        default='/workspace/home/rsna-miccai-challenge/train_labels_fold.csv')


    parser.add_argument('--path_data',
                        type=str,
                        default='../data/golby_data/iUS-T2-shape-reg-005mm-crop/')
    
    parser.add_argument('--seed',
                    type=int,
                    default=2)

    parser.add_argument('--batch_size',
                    type=int,
                    default=6)

    parser.add_argument('--max_features',
                    type=int,
                    default=64)

    parser.add_argument('--base_features',
                    type=int,
                    default=8)

    parser.add_argument('--pools',
                    type=int,
                    default=4)

    parser.add_argument('--epochs',
                    type=int,
                    default=1000)

    parser.add_argument('--temp',
                    type=float,
                    default=0.2)


    parser.add_argument('--spatial_shape',
                    type=int,
                    nargs="+",
                    default=SPATIAL_SHAPE)

    parser.add_argument('--modalities',
                    type=str,
                    nargs="+",
                    default=MODALITIES)

    parser.add_argument('--no_se', action='store_false', help='Squeeze and Excitation disabled')

    parser.add_argument('--no_res', action='store_false', help='Residual connection disabled')
    
    parser.add_argument('--save_images', action='store_true', help='Save images')
    
    parser.add_argument('--save_features', action='store_true', help='Save features')
    
    parser.add_argument('--save_samples', action='store_true', help='Save samples')



    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    main()


