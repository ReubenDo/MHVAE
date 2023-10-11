#  Unified Brain MR-Ultrasound Synthesis using Multi-Modal Hierarchical Representations

Public PyTorch implementation for our paper [Unified Brain MR-Ultrasound Synthesis using Multi-Modal Hierarchical Representations](https://arxiv.org/abs/2309.08747), 
which was accepted for presentation at [MICCAI 2023]([https://www.miccai2021.org](https://conferences.miccai.org/2023/en/)). 

If you find this code useful for your research, please cite the following paper:

```
@inproceedings{dorent2023unified,
  title={Unified Brain MR-Ultrasound Synthesis Using Multi-modal Hierarchical Representations},
  author={Dorent, Reuben and Haouchine, Nazim and Kogl, Fryderyk and Joutard, Samuel and Juvekar, Parikshit and Torio, Erickson and Golby, Alexandra J and Ourselin, Sebastien and Frisken, Sarah and Vercauteren, Tom and others},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={448--458},
  year={2023},
  organization={Springer}
}

```

## Method Overview
We introduce MHVAE, a deep hierarchical variational auto-encoder (VAE) that synthesizes missing images from various modalities. 

*Example of synthesis (first column: input; last column: target groundtruth image; other columns: synthetic images for different temperature.
![image1](https://github.com/ReubenDo/MHVAE/assets/17268715/64f3af8a-6da6-4c0b-acfa-934a6115ada6)



## Virtual Environment Setup

The code is implemented in Python 3.6 using the PyTorch library. 
Requirements:

 * Set up a virtual environment (e.g. conda or virtualenv) with Python >=3.6.9
 * Install all requirements using:
  
  ````pip install -r requirements.txt````
  

## Data

The data and annotations are publicly available on [TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=157288106).

## Running the code
`train.py` is the main file for training the models.

`inference.py` is the main file for running the inference:
 
## Using the code with your own data

If you want to use your own data, you just need to change the source and target paths, 
the splits and potentially the modality used.
