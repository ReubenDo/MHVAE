import os
import logging
import pandas as pd
import torch
import nibabel as nib  
import numpy as np


def create_logger(folder):
    """Create a logger to save logs."""
    compt = 0
    while os.path.exists(os.path.join(folder,f"logs_{compt}.txt")):
        compt+=1
    logname = os.path.join(folder,f"logs_{compt}.txt")
    
    logger = logging.getLogger()
    fileHandler = logging.FileHandler(logname, mode="w")
    consoleHandler = logging.StreamHandler()
    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)
    formatter = logging.Formatter("%(message)s")
    fileHandler.setFormatter(formatter)
    consoleHandler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    return logger 

def infinite_iterable(i):
    while True:
        yield from i

def poly_lr(epoch: int, max_epochs: int, initial_lr: float, min_lr: float=1e-5, exponent: float=0.9) -> float:
    return min_lr + (initial_lr - min_lr) * (1 - epoch / max_epochs)**exponent


def init_training_variables(model, netD, opt, logger, phase='training'):
    df_path = os.path.join(opt.model_dir,'log.csv')
    save_path = os.path.join(opt.model_dir, 'models', './CP_{}_{}.pth')
    
    if os.path.isfile(df_path): # If the training already started
        df = pd.read_csv(df_path, index_col=False)
        epoch = int(df.iloc[-1]['epoch']/10)*10
    else:
        epoch = 0
        
    if epoch>=10:
        best_epoch = epoch
        val_eval_criterion_MA = df.iloc[-1]['MA']
        best_val_eval_criterion_MA = df.iloc[-1]['best_MA']
        initial_lr = df.iloc[-1]['lr']
        model.load_state_dict(torch.load(save_path.format('main',epoch)))
        logger.info(f"Loading model from {save_path.format('main',epoch)}")
        try:
            for mod in opt.modalities:
                netD[mod].load_state_dict(torch.load(save_path.format(mod,epoch)))
                logger.info(f"Loading {mod} distcriminator from {save_path.format(mod,epoch)}")
        except Exception as e:
            logger.info(e)
    else: # If training from scratch
        df = pd.DataFrame(columns=['epoch','best_epoch', 'MA', 'best_MA', 'lr'])
        val_eval_criterion_MA = None
        best_epoch = 0
        epoch = 0
        best_val_eval_criterion_MA = 1e10
        initial_lr = opt.ini_lr
        # if not opt.load_net is None:
        #     model.load_state_dict(torch.load(opt.load_net))
        #     logger.info(f"Loading model from {opt.load_net}")
            
        
    return df, val_eval_criterion_MA, best_val_eval_criterion_MA, best_epoch, epoch, initial_lr



def save(pred, affine, path):
    pred = (255*(pred+1)/2).int()
    pred = pred.permute(2,3,0,1).squeeze().cpu().numpy()
    nib.Nifti1Image(pred, affine).to_filename(path)
    

def quantisize(img_data, levels=256, lower=0.0, upper=99.95):
    min_data = np.percentile(img_data, lower)
    max_data = np.percentile(img_data, upper)
    
    img_data[img_data<min_data] = min_data 
    img_data[img_data>max_data] = max_data
    img_data = (img_data-min_data) / (max_data - min_data+1e-8)
    img_data = np.digitize(img_data.squeeze(), np.arange(0,levels-1)/(levels-1)  ) 

    return img_data.astype(np.uint8)    

    
def save_feature(features, paths, affine):
    pred = np.stack([pred.permute(2,3,0,1).squeeze().cpu().numpy() for pred in features],0)
    out = []
    nb_channels = pred.shape[-1]
    nb_imgs = pred.shape[0]
    for c in range(nb_channels):
        out.append(quantisize(pred[...,c]))
    out = np.stack(out,-1)
    for i_image in range(nb_imgs): 
        nib.Nifti1Image(out[i_image,...], affine).to_filename(paths[i_image])
