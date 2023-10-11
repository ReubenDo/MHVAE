# Adapted from
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical, Normal

@torch.jit.script
def soft_clamp(x: torch.Tensor):
    return x.div(20.).tanh_()

@torch.jit.script
def soft_clamp_scale(x: torch.Tensor):
    return x.div(20.).tanh_()*0.0

@torch.jit.script
def soft_clamp5(x: torch.Tensor):
    return x.div(5.).tanh_().mul(5.)    #  5. * torch.tanh(x / 5.) <--> soft differentiable clamp between [-5, 5]

def one_hot(indices, depth, dim):
    indices = indices.unsqueeze(dim)
    size = list(indices.size())
    size[dim] = depth
    y_onehot = torch.zeros(size).cuda()
    y_onehot.zero_()
    y_onehot.scatter_(dim, indices, 1)

    return y_onehot

@torch.jit.script
def sample_normal_jit(mu, sigma):
    eps = mu.mul(0).normal_()
    z = eps.mul_(sigma).add_(mu)
    return z, eps

class NormalDecoder:
    def __init__(self, param, num_bits=8):
        B, C, H, W, D = param.size()
        self.num_c = C // 2
        mu = param[:, :self.num_c, :, :, :]                                 # B, 1, H, W, D
        log_sigma = param[:, self.num_c:, :, :, :]                          # B, 1, H, W, D
        self.dist = Normal(mu, log_sigma)

    def log_prob(self, samples):
        assert torch.max(samples) <= 1.0 and torch.min(samples) >= 0.0
        # convert samples to be in [-1, 1]
        samples = 2 * samples - 1.0

        return self.dist.log_p(samples)

    def sample(self, t=1.):
        x, _ = self.dist.sample()
        x = torch.clamp(x, -1, 1.)
        x = x / 2. + 0.5
        return x


class DiscLogistic:
    def __init__(self, param):
        B, C, H, W, D = param.size()
        self.num_c = C // 2
        self.means = param[:, :self.num_c, :, :, :]                              # B, 3, H, W, D
        self.log_scales = torch.clamp(param[:, self.num_c:, :, :, :], min=-8.0)  # B, 3, H, W, D

    def log_prob(self, samples):
        assert torch.max(samples) <= 1.0 and torch.min(samples) >= 0.0
        # convert samples to be in [-1, 1]
        samples = 2 * samples - 1.0

        B, C, H, W, D = samples.size()
        assert C == self.num_c

        centered = samples - self.means                                         # B, 3, H, W, D
        inv_stdv = torch.exp(- self.log_scales)
        plus_in = inv_stdv * (centered + 1. / 255.)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered - 1. / 255.)
        cdf_min = torch.sigmoid(min_in)
        log_cdf_plus = plus_in - F.softplus(plus_in)
        log_one_minus_cdf_min = - F.softplus(min_in)
        cdf_delta = cdf_plus - cdf_min
        mid_in = inv_stdv * centered
        log_pdf_mid = mid_in - self.log_scales - 2. * F.softplus(mid_in)

        log_prob_mid_safe = torch.where(cdf_delta > 1e-5,
                                        torch.log(torch.clamp(cdf_delta, min=1e-10)),
                                        log_pdf_mid - np.log(127.5))
        # woow the original implementation uses samples > 0.999, this ignores the largest possible pixel value (255)
        # which is mapped to 0.9922
        log_probs = torch.where(samples < -0.999, log_cdf_plus, torch.where(samples > 0.99, log_one_minus_cdf_min,
                                                                            log_prob_mid_safe))   # B, 3, H, W

        return log_probs

    def sample(self):
        u = torch.Tensor(self.means.size()).uniform_(1e-5, 1. - 1e-5).cuda()                        # B, 3, H, W
        x = self.means + torch.exp(self.log_scales) * (torch.log(u) - torch.log(1. - u))            # B, 3, H, W
        x = torch.clamp(x, -1, 1.)
        x = x / 2. + 0.5
        return x



class DiscMixLogistic:
    def __init__(self, param, num_mix=10, num_bits=8):
        B, C, H, W, D = param.size()
        self.num_mix = num_mix
        self.logit_probs = param[:, :num_mix, :, :, :]                                   # B, M, H, W, D
        l = param[:, num_mix:, :, :, :].view(B, 1, 2 * num_mix, H, W, D)                 # B, 1, 2 * M, H, W, D
        self.means = soft_clamp5(l[:, :, :num_mix, :, :, :])                                        # B, 1, M, H, W, D
        self.log_scales = torch.clamp(l[:, :, num_mix:2 * num_mix, :, :, :], min=-7.0)   # B, 1, M, H, W, D
        self.max_val = 2. ** num_bits - 1

    def log_prob(self, samples):
        assert torch.max(samples) <= 1.0 and torch.min(samples) >= 0.0
        # convert samples to be in [-1, 1]
        samples = 2 * samples - 1.0

        B, C, H, W, D = samples.size()
        assert C == 1, 'only single-channel images are considered.'

        samples = samples.unsqueeze(5)                                                  # B, 1, H, W, D
        samples = samples.expand(-1, -1, -1, -1, -1, self.num_mix).permute(0, 1, 5, 2, 3, 4)   # B, 1, M, H, W, D
        
        means = self.means                                  # B, 1, M, H, W
        centered = samples - means                          # B, 1, M, H, W

        inv_stdv = torch.exp(- self.log_scales)
        plus_in = inv_stdv * (centered + 1. / self.max_val)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered - 1. / self.max_val)
        cdf_min = torch.sigmoid(min_in)
        log_cdf_plus = plus_in - F.softplus(plus_in)
        log_one_minus_cdf_min = - F.softplus(min_in)
        cdf_delta = cdf_plus - cdf_min
        mid_in = inv_stdv * centered
        log_pdf_mid = mid_in - self.log_scales - 2. * F.softplus(mid_in)

        log_prob_mid_safe = torch.where(cdf_delta > 1e-5,
                                        torch.log(torch.clamp(cdf_delta, min=1e-10)),
                                        log_pdf_mid - np.log(self.max_val / 2))
        # the original implementation uses samples > 0.999, this ignores the largest possible pixel value (255)
        # which is mapped to 0.9922
        log_probs = torch.where(samples < -0.999, log_cdf_plus, torch.where(samples > 0.99, log_one_minus_cdf_min,
                                                                            log_prob_mid_safe))   # B, 1, M, H, W, D

        log_probs = torch.sum(log_probs, 1) + F.log_softmax(self.logit_probs, dim=1)  # B, M, H, W, D
        return torch.logsumexp(log_probs, dim=1)                                      # B, H, W, D

    def sample(self, t=0.7):
        B, C, H, W, D = self.logit_probs.size()
        gumbel = -torch.log(- torch.log(torch.Tensor(B, C, H, W, D).uniform_(1e-5, 1. - 1e-5).cuda()))  # B, M, H, W, D

        sel = one_hot(torch.argmax(self.logit_probs / t + gumbel, 1), self.num_mix, dim=1)          # B, M, H, W, D
        sel = sel.unsqueeze(1)                                                                 # B, 1, M, H, W, D

        # select logistic parameters
        means = torch.sum(self.means * sel, dim=2)                                             # B, 1, H, W, D
        log_scales = torch.sum(self.log_scales * sel, dim=2)                                   # B, 1, H, W, D
        
        # cells from logistic & clip to interval
        # we don't actually round to the nearest 8bit value when sampling
        u = torch.Tensor(B, 1, H, W, D).uniform_(1e-5, 1. - 1e-5).cuda()                        # B, 1, H, W, D
        x = means + torch.exp(log_scales) / t * (torch.log(u) - torch.log(1. - u))             # B, 1, H, W, D
        
        x = torch.clamp(x, -1, 1.)                                                # B, H, W, D
        x = x / 2. + 0.5
        return x

    def mean(self):
        sel = torch.softmax(self.logit_probs, dim=1)                                           # B, M, H, W, D
        sel = sel.unsqueeze(1)                                                                 # B, 1, M, H, W, D

        # select logistic parameters
        means = torch.sum(self.means * sel, dim=2)                                             # B, 1, H, W, D

        # we don't sample from logistic components, because of the linear dependencies, we use mean
        x = means                                                                              # B, 1, H, W, D
        x = torch.clamp(x, -1, 1.)                                                # B, H, W, D

        x = x / 2. + 0.5
        return x


class DiscMixLogistic2D:
    def __init__(self, param, num_mix=10, num_bits=8, min_logscales=-7.0):
        B, C, H, W = param.size()
        self.num_mix = num_mix
        self.logit_probs = param[:, :num_mix, :, :]                                   # B, M, H, W
        l = param[:, num_mix:, :, :].view(B, 1, 2 * num_mix, H, W)                 # B, 1, 2 * M, H, W
        self.means = soft_clamp(l[:, :, :num_mix, :, :])                                          # B, 1, M, H, W
        self.log_scales = torch.clamp(l[:, :, num_mix:2 * num_mix, :, :], min=min_logscales, max=-5.)   # B, 1, M, H, W
        self.max_val = 2. ** num_bits - 1

    def log_prob(self, samples):
        assert torch.max(samples) <= 1.0 and torch.min(samples) >= 0.0
        # convert samples to be in [-1, 1]
        samples = 2 * samples - 1.0

        B, C, H, W = samples.size()
        assert C == 1, 'only single-channel images are considered.'

        samples = samples.unsqueeze(4)                                                  # B, 1, H, W, 1
        samples = samples.expand(-1, -1, -1, -1, self.num_mix).permute(0, 1, 4, 2, 3)   # B, 1, M, H, W
        
        means = self.means                                  # B, 1, M, H, W
        centered = samples - means                          # B, 1, M, H, W

        inv_stdv = torch.exp(- self.log_scales)
        plus_in = inv_stdv * (centered + 1. / self.max_val)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered - 1. / self.max_val)
        cdf_min = torch.sigmoid(min_in)
        log_cdf_plus = plus_in - F.softplus(plus_in)
        log_one_minus_cdf_min = - F.softplus(min_in)
        cdf_delta = cdf_plus - cdf_min
        mid_in = inv_stdv * centered
        log_pdf_mid = mid_in - self.log_scales - 2. * F.softplus(mid_in)

        log_prob_mid_safe = torch.where(cdf_delta > 1e-5,
                                        torch.log(torch.clamp(cdf_delta, min=1e-10)),
                                        log_pdf_mid - np.log(self.max_val / 2))
        # the original implementation uses samples > 0.999, this ignores the largest possible pixel value (255)
        # which is mapped to 0.9922
        log_probs = torch.where(samples < -0.999, log_cdf_plus, torch.where(samples > 0.99, log_one_minus_cdf_min,
                                                                            log_prob_mid_safe))   # B, 1, M, H, W, D

        log_probs = torch.sum(log_probs, 1) + F.log_softmax(self.logit_probs, dim=1)  # B, M, H, W, D
        return torch.logsumexp(log_probs, dim=1)                                      # B, H, W, D

    def sample(self, t=0.7):
        B, C, H, W = self.logit_probs.size()
        gumbel = -torch.log(- torch.log(torch.Tensor(B, C, H, W).uniform_(1e-5, 1. - 1e-5).cuda()))  # B, M, H, W

        sel = one_hot(torch.argmax(self.logit_probs / t + gumbel, 1), self.num_mix, dim=1)          # B, M, H, W
        sel = sel.unsqueeze(1)                                                                 # B, 1, M, H, W

        # select logistic parameters
        means = torch.sum(self.means * sel, dim=2)                                             # B, 1, H, W
        log_scales = torch.sum(self.log_scales * sel, dim=2)                                   # B, 1, H, W
        
        # cells from logistic & clip to interval
        # we don't actually round to the nearest 8bit value when sampling
        u = torch.Tensor(B, 1, H, W).uniform_(1e-5, 1. - 1e-5).cuda()                        # B, 1, H, W
        x = means + torch.exp(log_scales) / t * (torch.log(u) - torch.log(1. - u))             # B, 1, H, W
        
        x = torch.clamp(x, -1, 1.)                                                # B, H, W, D
        x = x / 2. + 0.5
        return x

    def mean(self, postprocessed=True):
        sel = torch.softmax(self.logit_probs, dim=1)                                           # B, M, H, W
        sel = sel.unsqueeze(1)                                                                 # B, 1, M, H, W

        # select logistic parameters
        means = torch.sum(self.means * sel, dim=2)                                             # B, 1, H, W

        # we don't sample from logistic components, because of the linear dependencies, we use mean
        x = means                                                                              # B, 1, H, W
        if postprocessed:
            x = torch.clamp(x, -1, 1.)                                                # B, H, W, D

            x = x / 2. + 0.5
        return x



class DiscMixLogistic2DTest:
    def __init__(self, param, num_mix=10, num_bits=8, min_logscales=-7.0):
        B, C, H, W = param.size()
        self.num_mix = num_mix
        self.logit_probs = param[:, :num_mix, :, :]                                   # B, M, H, W
        print('-'*5)
        print(torch.exp(self.logit_probs).mean(0).mean(1).mean(1).std())
        print(torch.max(F.softmax(self.logit_probs, dim=1),1)[0].mean())
        l = param[:, num_mix:, :, :].view(B, 1, 2 * num_mix, H, W)                 # B, 1, 2 * M, H, W
        self.means = soft_clamp(l[:, :, :num_mix, :, :])                                          # B, 1, M, H, W
        self.log_scales = torch.clamp(l[:, :, num_mix:2 * num_mix, :, :], min=min_logscales, max=-5.)   # B, 1, M, H, W
        print(torch.exp(self.log_scales).mean())
        self.max_val = 2. ** num_bits - 1

    def log_prob(self, samples):
        assert torch.max(samples) <= 1.0 and torch.min(samples) >= 0.0
        # convert samples to be in [-1, 1]
        samples = 2 * samples - 1.0

        B, C, H, W = samples.size()
        assert C == 1, 'only single-channel images are considered.'

        samples = samples.unsqueeze(4)                                                  # B, 1, H, W, 1
        samples = samples.expand(-1, -1, -1, -1, self.num_mix).permute(0, 1, 4, 2, 3)   # B, 1, M, H, W
        
        means = self.means                                  # B, 1, M, H, W
        centered = samples - means                          # B, 1, M, H, W

        inv_stdv = torch.exp(- self.log_scales)
        plus_in = inv_stdv * (centered + 1. / self.max_val)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered - 1. / self.max_val)
        cdf_min = torch.sigmoid(min_in)
        log_cdf_plus = plus_in - F.softplus(plus_in)
        log_one_minus_cdf_min = - F.softplus(min_in)
        cdf_delta = cdf_plus - cdf_min
        mid_in = inv_stdv * centered
        log_pdf_mid = mid_in - self.log_scales - 2. * F.softplus(mid_in)

        log_prob_mid_safe = torch.where(cdf_delta > 1e-5,
                                        torch.log(torch.clamp(cdf_delta, min=1e-10)),
                                        log_pdf_mid - np.log(self.max_val / 2))
        # the original implementation uses samples > 0.999, this ignores the largest possible pixel value (255)
        # which is mapped to 0.9922
        log_probs = torch.where(samples < -0.999, log_cdf_plus, torch.where(samples > 0.99, log_one_minus_cdf_min,
                                                                            log_prob_mid_safe))   # B, 1, M, H, W, D

        log_probs = torch.sum(log_probs, 1) + F.log_softmax(self.logit_probs, dim=1)  # B, M, H, W, D
        return torch.logsumexp(log_probs, dim=1)                                      # B, H, W, D

    def sample(self, t=0.7):
        out = []
        for i in range(30):
            B, C, H, W = self.logit_probs.size()
            gumbel = -torch.log(- torch.log(torch.Tensor(B, C, H, W).uniform_(1e-5, 1. - 1e-5).cuda()))  # B, M, H, W

            sel = one_hot(torch.argmax(self.logit_probs / t + gumbel, 1), self.num_mix, dim=1)          # B, M, H, W
            #sel = one_hot(Categorical(F.softmax(self.logit_probs, dim=1).permute(0,2,3,1)).sample(), self.num_mix, dim=1)
            #sel = torch.argmax(self.logit_probs, 1)
            sel = sel.unsqueeze(1)                                                                 # B, 1, M, H, W

            # select logistic parameters
            means = torch.sum(self.means * sel, dim=2)                                             # B, 1, H, W
            log_scales = torch.sum(self.log_scales * sel, dim=2)                                   # B, 1, H, W
            
            # cells from logistic & clip to interval
            # we don't actually round to the nearest 8bit value when sampling
            u = torch.Tensor(B, 1, H, W).uniform_(1e-5, 1. - 1e-5).cuda()                        # B, 1, H, W
            x = means + torch.exp(log_scales) / t * (torch.log(u) - torch.log(1. - u))             # B, 1, H, W
            
            x = torch.clamp(x, -1, 1.)                                                # B, H, W, D
            x = x / 2. + 0.5
            out.append(x)
        out = torch.stack(out,0)
        out = out.mean(0)
        return out
    
    # def sample(self, t=100):
    #     sel = torch.softmax(self.logit_probs, dim=1)                                           # B, M, H, W
    #     sel = sel.unsqueeze(1)                                                                 # B, 1, M, H, W

    #     # select logistic parameters
    #     means = torch.sum(self.means * sel, dim=2)                                             # B, 1, H, W

    #     # we don't sample from logistic components, because of the linear dependencies, we use mean
    #     x = means                                                                              # B, 1, H, W
    #     x = torch.clamp(x, -1, 1.)                                                # B, H, W, D

    #     x = x / 2. + 0.5
    #     return x

    def mean(self,postprocessed=True):
        sel = torch.softmax(self.logit_probs, dim=1)                                           # B, M, H, W
        sel = sel.unsqueeze(1)                                                                 # B, 1, M, H, W

        # select logistic parameters
        means = torch.sum(self.means * sel, dim=2)                                             # B, 1, H, W

        # we don't sample from logistic components, because of the linear dependencies, we use mean
        x = means                                                                              # B, 1, H, W
        if postprocessed:
            x = torch.clamp(x, -1, 1.)                                                # B, H, W, D

            x = x / 2. + 0.5
        return x
