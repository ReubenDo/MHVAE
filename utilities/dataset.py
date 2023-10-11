import os
import numpy as np
import random
import nibabel as nib

import torch
from torch.utils.data import Dataset


class DatasetRDO(Dataset):
    def __init__(self, paths, mode='training'):
        self.mode = mode
        self.paths = paths
        lenghts = [len(k) for k in paths.values()]
        assert all(x == lenghts[0] for x in lenghts)
        self.nb_scans = lenghts[0]
        print(mode, self.nb_scans)
        
    def preprocess(self, x, k=[0,0,0]):
        if self.mode=='training':
            if k[0]==1:
                x = x[::-1, :, :]
            if k[1]==1:
                x = x[:, ::-1, :]
            if k[2]==1:
                x = x[:, :, ::-1]
            x = x.copy()
        return x

    def __getitem__(self, index):
        output = dict()
        k = [random.randint(0,1), 0, 0]
        for mod in self.paths.keys():
            img = nib.load(self.paths[mod][index])
            affine = img.affine.squeeze()
            img = self.preprocess(img.get_fdata().squeeze(), k)
            output[mod] = torch.from_numpy(np.expand_dims(img, axis=0))
            output[mod+'_affine'] = torch.from_numpy(np.expand_dims(affine, axis=0))
            output[mod+'_name'] = os.path.basename(self.paths[mod][index].replace('.nii.gz',''))
        return output

    def __len__(self):
        return self.nb_scans
    
