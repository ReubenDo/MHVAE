#   CODE ADAPTED FROM: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/network_architecture/generic_UNet.py

from copy import deepcopy
from torch import nn
import torch
import numpy as np
import torch.nn.functional
from torch.distributions import Normal
from network.blocks import BlockEncoder, AsFeatureMap_down, AsFeatureMap_up, BlockDecoder, BlockFinalImg, Upsample, BlockQ


def soft_clamp(x: torch.Tensor, v: int=10):
    return x.div(v).tanh_().mul(v)

def soft_clamp_img(x: torch.Tensor):
    return (x.div(5).tanh_() + 1 ) / 2 

class MHVAE2D(nn.Module):

    def __init__(
        self, 
        modalities, 
        base_num_features, 
        num_pool,
        num_img=4,
        num_feat_img=1,
        weightInitializer=None,
        original_shape=(160,192,144),
        max_features=64,
        with_residual=False,
        with_se=True,
        logger=None,
        return_cat=True,
        last_act='tanh'):
        """

        """
        super(MHVAE2D, self).__init__()

        self.weightInitializer = weightInitializer
        self.max_features = max_features
        self.num_feat_img = num_feat_img
        self.logger = logger
        self.return_cat = return_cat
        self.last_act = last_act
        self.modalities = modalities

        min_shape = [int(k/(2**num_pool)) for k in original_shape]  
        down_strides = [(2,2)]*(num_pool)
        up_kernel = down_strides

        nfeat_input = 1 
        nfeat_output = base_num_features
        self.first_conv = dict()
        # Layers for the modality-specific embeddings
        for mod in modalities:
            self.first_conv[mod] = nn.Conv2d(
                        in_channels=nfeat_input, 
                        out_channels=nfeat_output,
                        kernel_size=3,
                        stride=1, 
                        padding=1,
                        bias=True)

        self.conv_blocks_context = {mod:[] for mod in modalities}
        self.td =  {mod:[] for mod in modalities}
        for mod in modalities:
            nfeat_output = base_num_features
            for d in range(num_pool):
                nfeat_input = nfeat_output
                nfeat_output = min(2 * nfeat_output, max_features)
                self.conv_blocks_context[mod].append(BlockEncoder(nfeat_input, residual=with_residual, with_se=with_se))
                self.td[mod].append(nn.Conv2d(nfeat_input, nfeat_output, kernel_size=3, stride=down_strides[d], padding=1))
            
            self.conv_blocks_context[mod].append(BlockEncoder(nfeat_output, residual=with_residual, with_se=with_se))


        # Going from a feature volume to a feature vector (e.g (3,3,3)-->(1))
        self.bottleneck_down = AsFeatureMap_down(input_shape=[nfeat_output,]+min_shape, target_dim=4*max_features)
        # Going from a feature vector to a feature volume (e.g. (1)-->(3,3,3)) 
        self.bottleneck_up = AsFeatureMap_up(input_dim=2*max_features, target_shape=[max_features,]+min_shape)

        # Layers for the approximate posterior + prior
        nfeat_latent = max_features 
        self.tu = []
        self.conv_blocks_localization = []
        self.qz = []
        self.pz = []
        for u in np.arange(num_pool)[::-1]:
            nfeatures_from_skip = self.conv_blocks_context[modalities[0]][u].output_channels

            n_features_after_tu_and_concat = 2*nfeatures_from_skip 
        
            self.tu.append(
                    Upsample(n_channels=nfeat_latent, n_out=nfeatures_from_skip, scale_factor=up_kernel[u], mode='bilinear')
                    )
            
            self.conv_blocks_localization.append(BlockDecoder(nfeatures_from_skip, residual=with_residual, with_se=with_se))

            self.qz.append(BlockQ(n_features_after_tu_and_concat, nfeatures_from_skip, nfeatures_from_skip))
            self.pz.append(nn.utils.weight_norm(nn.Conv2d(nfeatures_from_skip, nfeatures_from_skip, 1, 1, 0, 1, 1, False), dim=0, name='weight'))

            nfeat_latent = nfeatures_from_skip // 2

        self.final_blocks = [BlockFinalImg(nfeat_latent, num_feat_img, last_act) for i in range(num_img)]
        
        # register all modules properly
        self.first_conv = nn.ModuleDict(self.first_conv)
        for mod in modalities:
            self.conv_blocks_context[mod] = nn.ModuleList(self.conv_blocks_context[mod])
            self.td[mod] = nn.ModuleList(self.td[mod])
        self.conv_blocks_context = nn.ModuleDict(self.conv_blocks_context)
        self.td = nn.ModuleDict(self.td)
        
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.tu = nn.ModuleList(self.tu)

        self.qz = nn.ModuleList(self.qz)
        self.pz = nn.ModuleList(self.pz)

        self.final_blocks = nn.ModuleList(self.final_blocks)
        
        self.nb_latent = len(self.conv_blocks_context[modalities[0]])
        
        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)

    def create_encodings(self, x):
        skips = {mod:[] for mod in x.keys()}
            
        # Encode each modality independtly
        for mod in x.keys():
            x[mod] = self.first_conv[mod](x[mod])
            for d in range(self.nb_latent - 1):
                x[mod] = self.conv_blocks_context[mod][d](x[mod])
                skips[mod].append(x[mod])
                x[mod] = self.td[mod][d](x[mod])
        
            x[mod] = self.conv_blocks_context[mod][-1](x[mod])
            
        return x, skips
    
    def compute_marginal(self, mu_q_res, log_sigma_q_res, mu_p, inv_sigma_p, temp=1.0):
        inv_sigma_q_res = torch.exp(-log_sigma_q_res)
        
        sigma_q = 1 / (inv_sigma_q_res + inv_sigma_p + 1e-3)
        mu_q = sigma_q * (inv_sigma_q_res * mu_q_res + inv_sigma_p * mu_p)

        return Normal(mu_q, temp*sigma_q)
    
    def compute_full(self, res_params, prior, temp):
        mu = prior.loc / prior.scale
        inv_sigma = 1 / prior.scale
        
        for mod in res_params.keys():
            mu+= res_params[mod]['loc'] * torch.exp(-res_params[mod]['logscale_res'])
            inv_sigma += torch.exp(-res_params[mod]['logscale_res'])
        mu /= inv_sigma
        sigma = 1 / inv_sigma

        return Normal(mu, temp*sigma)
        

    def forward(self, x, temp=1, return_feat=False, verbose=False):
        modalities = list(x.keys()) # Corresponds to \pi in paper
        
        # Create embeddings for injecting info from (x_i)
        x, skips = self.create_encodings(x)
        
        # Initialization of distributions and their parameters
        distribs = {f'z{i+1}':dict() for i in range(self.nb_latent)}
        res_params = {f'z{i+1}':dict() for i in range(self.nb_latent)}
        
        # Computing q(z_L|x_i) 
        z_name = 'z{}'.format(self.nb_latent)
        for mod in modalities:
            mu_zl_q_res_mod, logvar_zl_q_res_mod = self.bottleneck_down(x[mod]).chunk(2, dim=1)
            mu_zl_q_res_mod = soft_clamp(mu_zl_q_res_mod)
            logvar_zl_q_res_mod = soft_clamp(logvar_zl_q_res_mod)
            res_params[z_name][mod] = {'loc':mu_zl_q_res_mod, 'logscale_res': logvar_zl_q_res_mod}
            distribs[z_name][mod] = self.compute_marginal(mu_zl_q_res_mod, logvar_zl_q_res_mod, 0, 1, temp) # prior is N(0,I)
        
        # p(z_L)
        mu_zl_p = torch.zeros_like(mu_zl_q_res_mod)
        sigma_zl_p = torch.ones_like(logvar_zl_q_res_mod)
        distribs[z_name]['prior'] = Normal(mu_zl_p, sigma_zl_p)
            
        # Approximate posterior for q(z_L|x_{\pi}) = p(z_L) \prod_{i\in\pi} q(z_L|x_i)
        distribs[z_name]['full'] = self.compute_full(res_params[z_name], distribs[z_name]['prior'], temp)
        
        # Sampling zL_q from q(z_L|x_{\pi})
        zl_q = distribs[z_name]['full'].rsample()
        if verbose:
            self.logger.info(f"Shape {z_name}: {zl_q.size()}")
        
        # Computing KLs
        kls = dict()
        for mod in modalities + ['prior']:
            kls[mod] = []
            kl = distribs[z_name]['full'].log_prob(zl_q) - distribs[z_name][mod].log_prob(zl_q)
            kls[mod].append(kl.sum())
        
        # Creating initial feature volume for z_{L-1}
        zl_q_up = self.bottleneck_up(zl_q)
        z_full = {z_name:zl_q_up}

        for i in range(self.nb_latent - 1): 
            z_name = 'z{}'.format(self.nb_latent-(i+1)) # = z^{l-1}
            
            # Creating feature volume for z_{l-1}
            z_ip1 = z_full['z{}'.format(self.nb_latent-i)]
            x = self.tu[i](z_ip1)
            x = self.conv_blocks_localization[i](x)
            if verbose:
                self.logger.info(f"Shape feature volume for {z_name}: {x.size()}")
            
            # Prior p(z_{l-1}|z_l)
            mu_zi_p, logvar_zi_p = self.pz[i](x).chunk(2, dim=1)
            mu_zi_p = soft_clamp(mu_zi_p)
            logvar_zi_p = soft_clamp(logvar_zi_p)
            distribs[z_name]['prior'] = Normal(mu_zi_p, torch.exp(logvar_zi_p))
            
            # Computing  q(z_{l-1}|x_i,z_l) 
            for mod in modalities:
                # Merging embedding from z_{l-1} and x_i
                x_q = torch.cat((x, skips[mod][-(i + 1)]), dim=1)
                mu_zi_q_res_mod, logvar_zi_q_res_mod = self.qz[i](x_q).chunk(2, dim=1)
                mu_zi_q_res_mod = soft_clamp(mu_zi_q_res_mod)
                logvar_zi_q_res_mod = soft_clamp(logvar_zi_q_res_mod)
                res_params[z_name][mod] = {'loc':mu_zi_q_res_mod, 'logscale_res': logvar_zi_q_res_mod}
                distribs[z_name][mod] = self.compute_marginal(mu_zi_q_res_mod, logvar_zi_q_res_mod, mu_zi_p, torch.exp(-logvar_zi_p), temp) 
            
            # Approximate posterior for q(z_{l-1}|x_{\pi}, z_l) = p(z_{l-1}|z_l) \prod_{i\in\pi} q(z_{l-1}|x_i,z_l)
            distribs[z_name]['full'] = self.compute_full(res_params[z_name], distribs[z_name]['prior'], temp)
        
            # Sampling z_{l-1}
            zi_q = distribs[z_name]['full'].rsample()
            if verbose:
                self.logger.info(f"Shape {z_name}: {zi_q.size()}")

            # Computing KLs
            for mod in modalities + ['prior']:
                kl = distribs[z_name]['full'].log_prob(zi_q) - distribs[z_name][mod].log_prob(zi_q)
                kls[mod].append(kl.sum())
            z_full[z_name] = zi_q            

        output_img = [fblock(z_full['z1']) for fblock in self.final_blocks]
        
        if self.return_cat:
            output_img = torch.cat(output_img, 1)

        if return_feat:
            return  output_img, kls, z_full['z1']
        else:
            return  output_img, kls        
    
       

    def sample(self, batch_size, temp=0.7):
        
        # Prior distribution for z_L
        mu = torch.zeros((batch_size,2*self.max_features)).cuda()
        sigma = torch.ones((batch_size,2*self.max_features)).cuda()
        p_zl = Normal(mu, temp*sigma)
        
        # Sample from p(z_L)
        zl_p = p_zl.sample()
        zl_p_up = self.bottleneck_up(zl_p)

        z_full = {'z{}'.format(self.nb_latent):zl_p_up}

        for i in range(self.nb_latent - 1):
            z_name = 'z{}'.format(self.nb_latent-(i+1))
            
            # Creating feature volume for z_{l-1}
            z_ip1 = z_full['z{}'.format(self.nb_latent-i)]
            x = self.tu[i](z_ip1)
            x = self.conv_blocks_localization[i](x)

            # Prior p(z_{l-1}|z_l)
            mu_zi_p, logvar_zi_p = self.pz[i](x).chunk(2, dim=1)
            mu_zi_p = soft_clamp(mu_zi_p)
            logvar_zi_p = soft_clamp(logvar_zi_p)
            var_zi_p = torch.exp(logvar_zi_p)
            p_zi =  Normal(mu_zi_p, temp*var_zi_p)
            
            # Sampling z_{l-1}
            zi_p = p_zi.sample()
            z_full[z_name] = zi_p

        if self.return_cat:
            return torch.cat([fblock(z_full['z1']) for fblock in self.final_blocks], 1)
        else:
            return [fblock(z_full['z1']) for fblock in self.final_blocks]

