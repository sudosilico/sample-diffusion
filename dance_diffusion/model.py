import torch
from torch import nn
from contextlib import nullcontext
from typing import Callable
from audio_diffusion.models import DiffusionAttnUnet1D


class ModelBase():
    def __init__(self):
        self.device: torch.device = None
        self.module: nn.Module = None
        self.chunk_size: int = 65536
        self.sample_rate: int = 48000
        
    def update(
        self,
        device: torch.device,
        module: nn.Module,
        chunk_size: int,
        sample_rate: int
    ):
        self.device = device
        self.module = module
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate


class GlobalArgs(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class DanceDiffusionInference(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.diffusion_ema = DiffusionAttnUnet1D(GlobalArgs(**kwargs), n_attn_layers=4)
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)


class DanceDiffusionModel(ModelBase):
    def __init__(
        self, 
        device: torch.device,
        chunk_size: int = 65536,
        sample_rate: int = 48000,
    ):
        super().__init__()

        self.module: DanceDiffusionInference = None
        self.device: torch.device = device
        self.model_path: str = None
        self.raw_function: Callable = None
        self.v_function: Callable = None
        
        self.chunk_size: int = chunk_size
        self.sample_rate: int = sample_rate
        
    def load(self, model_path: str, chunk_size: int = 65536, sample_rate: int = 48000):
        
        if self.model_path == model_path:
            return

        self.module = DanceDiffusionInference(
            sample_size=chunk_size,
            sample_rate=sample_rate,
            latent_dim=0,
        )
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate

        self.model_path = model_path
        
        self.module.load_state_dict(
            torch.load(model_path, map_location=self.device)["state_dict"], 
            strict=False,
        )
        
        # with torch.no_grad():
        self.module = self.module.requires_grad_(False).to(self.device)
        
        self.raw_function = self.module.diffusion_ema

        def autocast_model_fn(*args, **kwargs):
            precision_scope = torch.autocast

            # TODO: look into if autocast will work on cpu
            if self.device.type in ['mps', 'cpu']:
                precision_scope = nullcontext
            with precision_scope(self.device.type):
                m = self.raw_function(*args, **kwargs).float() 
                return m

        self.v_function = autocast_model_fn
        
        super().update(self.device, self.module, chunk_size, sample_rate)
        # add self.k_function = VWrapper(self.raw_function)
