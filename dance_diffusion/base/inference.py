import torch
import enum
import contextlib
from contextlib import nullcontext

from .model import ModelWrapperBase

class InferenceBase():
    def __init__(
        self,
        device_accelerator: torch.device,
        device_offload: torch.device,
        optimize_memory_use: bool,
        use_autocast: bool,
        model: ModelWrapperBase
    ):
        self.device_accelerator = device_accelerator
        self.device_offload = device_offload if(optimize_memory_use==True) else None
        self.optimize_memory_use = optimize_memory_use
        self.use_autocast = use_autocast
        self.model = model
        # self.generator = torch.Generator(device_accelerator if (device_accelerator.type != 'mps') else torch.device('cpu'))
        self.generator = torch.Generator(device_accelerator)

        
    def set_device_accelerator(
        self,
        device: torch.device = None
    ):
        self.device_accelerator = device
    
    def get_device_accelerator(
        self
    ) -> torch.device:
        return self.device_accelerator
        
    def set_model(
        self,
        model: ModelWrapperBase = None
    ):
        self.model = model
    
    def get_model(
        self
    ) -> ModelWrapperBase:
        return self.model

    def autocast_context(self):
        match self.device_accelerator.type:
            case 'cuda':
                return torch.cuda.amp.autocast()
            case 'cpu':
                return torch.cpu.amp.autocast()
            case 'mps':
                return nullcontext()
            case _:
                return torch.autocast(self.device_accelerator.type, dtype=torch.float32)

    @contextlib.contextmanager
    def offload_context(self, model):
        """
            Used by inference implementations, this context manager moves the
            passed model to the inference's `device_accelerator` device on enter,
            and then returns it to the `device_offload` device on exit.

            It also wraps the `inference.autocast_context()` context.
        """

        autocast = self.autocast_context() if self.use_autocast else nullcontext()
        with autocast:
            if self.optimize_memory_use:
                model.to(self.device_accelerator)

            yield None

            if self.optimize_memory_use:
                model.to(self.device_offload)