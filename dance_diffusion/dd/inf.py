import torch
from typing import Callable

from dance_diffusion.base.model import ModelWrapperBase
from dance_diffusion.base.inference import InferenceBase

class DDInference(InferenceBase):
    
    def __init__(
        self,
        device: torch.device = None,
        model: ModelWrapperBase = None
    ):
        super().__init__(device, model)
        
    def generate(
        self,
        seed: int = None,
        batch_size: int = None,
        callback: Callable = None,
        **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError
    
    def generate_variation(
        self,
        seed: int = None,
        batch_size: int = None,
        audio_source: torch.Tensor = None,
        noise_level: float = None,
        callback: Callable = None,
        **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError
    
    def generate_interpolation(
        self,
        seed: int = None,
        batch_size: int = None,
        interpolations: int = None,
        audio_source: torch.Tensor = None,
        audio_target: torch.Tensor = None,
        noise_level: float = None,
        callback: Callable = None,
        **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError
    
    def generate_extension(
        self,
        seed: int = None,
        batch_size: int = None,
        audio_source: torch.Tensor = None,
        callback: Callable = None,
        **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError
    
    def generate_inpainting(
        self,
        seed: int = None,
        batch_size: int = None,
        audio_source: torch.Tensor = None,
        mask: torch.Tensor = None,
        callback: Callable = None,
        **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError
