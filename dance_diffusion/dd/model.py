        # TODO: handle tensor_result, return Response
import torch
from torch import nn
from typing import Callable

from .ddattnunet import DiffusionAttnUnet1D
from dance_diffusion.base.model import ModelWrapperBase


class GlobalArgs(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class DanceDiffusionInference(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.diffusion_ema = DiffusionAttnUnet1D(GlobalArgs(**kwargs), n_attn_layers=4)


class DDModelWrapper(ModelWrapperBase):
    def __init__(self):
        
        super().__init__()
        
        self.module:DanceDiffusionInference = None
        self.model:Callable = None
        
    def load(
        self,
        path:str,
        device_accelerator:torch.device,
        optimize_memory_use:bool=False,
        chunk_size:int=65536,
        sample_rate:int=48000
    ):

        self.path = path
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        
        self.module = DanceDiffusionInference(
            sample_size=chunk_size,
            sample_rate=sample_rate,
            latent_dim=0,
        )
        
        self.module.load_state_dict(
            torch.load(path)["state_dict"], 
            strict=False
        )
        self.module.eval().requires_grad_(False)
        
        self.model = self.module.diffusion_ema if (optimize_memory_use) else self.module.diffusion_ema.to(device_accelerator)
        