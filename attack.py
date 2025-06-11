import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms as T
from tools import *
import random

img_min = 0.0
img_max = 1.0

class DSA:
    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1., num_scale=5, num_block=3,
                 targeted=False, random_start=False,mix_low=0.7, mix_high=0.9,num_aug=4,
                 norm='linfty', loss='crossentropy', device=None, attack='DSA',num_global=15,dataset=None,**kwargs):

        self.attack = attack
        self.device = device
        self.model = load_model(model_name, self.device)
        self.epsilon = epsilon
        self.targeted = targeted
        self.random_start = random_start
        self.norm = norm
        self.loss = self.loss_function(loss)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.num_scale = num_scale
        self.num_global = num_global
        self.num_block = num_block
        self.dataset = dataset
        self.mix_low = mix_low
        self.mix_high = mix_high

    def forward(self, data, label, **kwargs):
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        delta = self.init_delta(data)
        momentum = 0
        for _ in range(self.epoch):
            x_transformed_list = self.transform(data + delta)
            gradients = []
            momentum = self.get_momentum(mean_grad, momentum)
            delta = self.update_delta(delta, data, momentum, self.alpha)
            torch.cuda.empty_cache()
        return delta.detach()

##### Core functions will be updated once the paper has been accepted for publication. #####

    def __call__(self, *input, **kwargs):
        self.model.eval()
        return self.forward(*input, **kwargs)

    def transform(self, x, ** kwargs):

        transformed = []
        B = x.size(0)

        for _ in range(self.num_scale):
            transformed.append(self.blocktransform(x))

        if self.dataset is not None:
            for _ in range(self.num_global):

                random_imgs = self.get_random_batch(B)
                mixed = self.mix_images(x, random_imgs)
                x_aug = mixed.clone()
                for _ in range(self.num_aug):
                    op = np.random.choice(self.global_ops)
                    x_aug = op(x_aug)
                transformed.append(x_aug)
                
                torch.cuda.empty_cache()

        return transformed


    def loss_function(self, loss):
        if loss == 'crossentropy':
            return nn.CrossEntropyLoss(reduction='sum')
        else:
            raise Exception(f"Unsupported loss {loss}")

    def init_delta(self, data):
        delta = torch.zeros_like(data).to(self.device)
        if self.random_start:
            if self.norm == 'linfty':
                delta.uniform_(-self.epsilon, self.epsilon)
            else:
                delta.normal_(-self.epsilon, self.epsilon)
                d_flat = delta.view(delta.size(0), -1)
                n = d_flat.norm(p=2, dim=-1).view(delta.size(0), 1, 1, 1)
                r = torch.zeros_like(data).uniform_(0, 1).to(self.device)
                delta *= r / n * self.epsilon
            delta = clamp(delta, img_min - data, img_max - data)
        delta.requires_grad = True
        return delta

    def update_delta(self, delta, data, grad, alpha):
    
        if self.norm == 'linfty':
            delta_new = torch.clamp(delta + alpha * grad.sign(), -self.epsilon, self.epsilon)
        else:
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1)
            scaled_grad = grad / (grad_norm + 1e-20)
            delta_new = (delta + scaled_grad * alpha).view(delta.size(0), -1).renorm(p=2, dim=1,maxnorm=self.epsilon).view_as(delta)

        delta_new = clamp(delta_new, img_min - data, img_max - data)

        delta.data = delta_new.data
        return delta

    def get_logits(self, x):  #
        return self.model(x)

    def get_loss(self, logits, label):
        return -self.loss(logits, label) if self.targeted else self.loss(logits, label)

    def get_grad(self, loss, delta):
        return torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]

    def get_momentum(self, grad, momentum):
        return momentum * self.decay + grad / (grad.abs().mean(dim=(1, 2, 3), keepdim=True))
