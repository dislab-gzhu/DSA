import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import os
from PIL import Image
from torchvision import transforms
import timm
from dataload import *


def clamp(x, x_min, x_max):
    return torch.min(torch.max(x, x_min), x_max)

def save_images(adversaries, filenames, output_dir):
    adversaries = (adversaries.detach().permute((0, 2, 3, 1)).cpu().numpy() * 255).astype(np.uint8)
    for i, filename in enumerate(filenames):
        Image.fromarray(adversaries[i]).save(os.path.join(output_dir, filename))

def wrap_model(model):
    model_name = model.__class__.__name__
    Resize = 224
    if hasattr(model, 'default_cfg'):
        """timm.models"""
        mean = model.default_cfg['mean']
        std = model.default_cfg['std']
    else:
        """torchvision.models"""
        if 'Inc' in model_name:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            Resize = 299
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            Resize = 224

    PreprocessModel = PreprocessingModel(Resize, mean, std)
    return torch.nn.Sequential(PreprocessModel, model) 

def load_model(model_name,device):
    def load_single_model(model_name):
        if model_name in models.__dict__.keys():
            print(f'=> Loading model {model_name} from torchvision.models')
            model = models.__dict__[model_name](weights="DEFAULT")
        elif model_name in timm.list_models():
            print(f'=> Loading model {model_name} from timm.models')
            model = timm.create_model(model_name, pretrained=True)
        else:
            raise ValueError(f'Model {model_name} not supported')
        return wrap_model(model.eval().to(device))
    
    return load_single_model(model_name)

class NormalizeLayer(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeLayer, self).__init__()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def forward(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

class PreprocessingModel(nn.Module):
    def __init__(self, resize, mean, std):
        super(PreprocessingModel, self).__init__()
        self.resize = transforms.Resize(resize,antialias=True)
        self.normalize = transforms.Normalize(mean, std)

    def forward(self, x):
        return self.normalize(self.resize(x))
