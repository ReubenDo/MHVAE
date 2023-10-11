from typing import Dict, Callable

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import torch.nn as nn


class AdversarialLoss(_Loss):
    def __init__(
        self,
        is_discriminator: bool = True,
        weight=None,
        size_average: bool = None,
        reduce: bool = None,
        reduction: str = "mean",
    ):
        super(AdversarialLoss, self).__init__(size_average, reduce, reduction)


        self.is_discriminator = is_discriminator
        self._weight = weight


    def forward(
        self, logits_fake: torch.Tensor, logits_real: torch.Tensor = None
    ) -> torch.Tensor:
        logits_fake = logits_fake.float()

        loss_fake = self.loss_function(
            logits_fake, False if self.is_discriminator else True
        )

        loss = loss_fake

        if self.is_discriminator:
            logits_real = logits_real.float()
            loss_real = self.loss_function(logits_real, True)
            
            loss = 0.5 * (loss + loss_real)

        loss = self._weight * loss

        return loss

    def get_weight(self) -> float:
        return self._weight

    def set_weight(self, weight: float) -> float:
        self._weight = weight

        return self.get_weight()

    def loss_function(self, logit: torch.Tensor, is_real: bool) -> torch.Tensor:
        # An equivalent explicit implementation would be
        #     loss_real = torch.mean((logits_real - 1) ** 2)
        #     loss_fake = torch.mean(logits_fake ** 2)
        return torch.mean((logit - (1 if is_real else 0)) ** 2)
    
    

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode: str, target_real_label: float=1.0, target_fake_label: float=0.0) -> None:
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss(reduction='mean')
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss(reduction='mean')
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss