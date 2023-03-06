import torch
import enum

from dataclasses import dataclass
from typing import Callable

from .base.model import ModelWrapperBase
from .base.inference import InferenceBase
from .dd.model import DDModelWrapper
from .dd.inference import DDInference

from diffusion_library.sampler import SamplerBase, DDPM, DDIM, IPLMS
from diffusion_library.scheduler import SchedulerBase, LinearSchedule, DDPMSchedule, SplicedDDPMCosineSchedule, LogSchedule, CrashSchedule

    
class RequestType(str, enum.Enum):
    Generation = "Generation"
    Variation = "Variation"
    Interpolation = "Interpolation"
    Inpainting = "Inpainting"
    Extension = "Extension"

class SamplerType(str, enum.Enum):
    DDPM = "DDPM"
    DDIM = "DDIM"
    IPLMS = "IPLMS"

class ModelType(str, enum.Enum):
    DD = "DD"

class SchedulerType(str, enum.Enum):
    LinearSchedule = "LinearSchedule"
    DDPMSchedule = "DDPMSchedule"
    SplicedDDPMCosineSchedule = "SplicedDDPMCosineSchedule"
    LogSchedule = "LogSchedule"
    CrashSchedule = "CrashSchedule"


class SchedulerArgs:
    def __init__(self, **kwargs):
        pass


class Request:
    def __init__(
        self,
        request_type: RequestType,
        model_path: str,
        model_type: ModelType,
        model_chunk_size: int,
        model_sample_rate: int,
        **kwargs
    ):
        self.request_type = request_type
        self.model_path = model_path
        self.model_type = model_type
        self.model_chunk_size = model_chunk_size
        self.model_sample_rate = model_sample_rate
        self.kwargs = kwargs


class Response:
    def __init__(
        self,
        result: torch.Tensor
    ):
        self.result = result


class RequestHandler:
    def __init__(
        self, 
        device_accelerator: torch.device, 
        device_offload: torch.device = None, 
        optimize_memory_use: bool = False,
        use_autocast: bool = True
    ):
        self.device_accelerator = device_accelerator
        self.device_offload = device_offload
        
        self.model_wrapper: ModelWrapperBase = None
        self.inference: InferenceBase = None 
        
        self.optimize_memory_use = optimize_memory_use
        self.use_autocast = use_autocast
        
    def process_request(
        self,
        request: Request,
        callback: Callable = None
    ) -> Response:
        # load the model from the request if it's not already loaded
        if (self.model_wrapper == None or request.model_path != self.model_wrapper.path): 
            self.load_model(
                request.model_type, 
                request.model_path, 
                request.model_chunk_size,
                request.model_sample_rate
            )
        
        match request.request_type:
            case RequestType.Generation:
                tensor_result = self.handle_generation(request, callback)
                    
            case RequestType.Variation:
                tensor_result = self.handle_variation(request, callback)
            
            case RequestType.Interpolation:
                tensor_result = self.handle_interpolation(request, callback)
            
            case RequestType.Inpainting:
                tensor_result = self.handle_inpainting(request, callback)

            case RequestType.Extension:
                tensor_result =  self.handle_extension(request, callback)
            
            case _:
                raise ValueError("Unexpected RequestType in process_request")

        return Response(tensor_result)

    def load_model(self, model_type, model_path, chunk_size, sample_rate):
        match model_type:
            case ModelType.DD:
                self.model_wrapper = DDModelWrapper()
                self.model_wrapper.load(
                    model_path,
                    self.device_accelerator,
                    self.optimize_memory_use,
                    chunk_size,
                    sample_rate
                )
                self.inference = DDInference(
                    self.device_accelerator,
                    self.device_offload,
                    self.optimize_memory_use,
                    self.use_autocast,
                    self.model_wrapper
                )
                
            case _:
                raise ValueError("Unexpected ModelType in load_model")

    def handle_generation(self, request: Request, callback: Callable) -> Response:
        match request.model_type:
            case ModelType.DD:
                return self.inference.generate(
                    callback=callback,
                    scheduler=self.create_scheduler(request.kwargs['scheduler_type']),
                    sampler=self.create_sampler(request.kwargs['sampler_type']),
                    **request.kwargs
                )
                
            case _:
                raise ValueError("Unexpected ModelType in handle_generation")

    def handle_variation(self, request: Request, callback: Callable) -> torch.Tensor:
        match request.model_type:
            case ModelType.DD:
                return self.inference.generate_variation(
                    callback=callback,
                    scheduler=self.create_scheduler(request.kwargs.get("scheduler_type")),
                    sampler=self.create_sampler(request.kwargs.get("sampler_type")),
                    **request.kwargs,
                )

            case _:
                raise ValueError("Unexpected ModelType in handle_variation")

    def handle_interpolation(self, request: Request, callback: Callable) -> torch.Tensor:
        match request.model_type:
            case ModelType.DD:
                return self.inference.generate_interpolation(
                    callback=callback,
                    scheduler=self.create_scheduler(request.kwargs.get("scheduler_type")),
                    sampler=self.create_sampler(request.kwargs.get("sampler_type")),
                    **request.kwargs,
                )
                
            case _:
                raise ValueError("Unexpected ModelType in handle_interpolation")
        
    def handle_inpainting(self, request: Request, callback: Callable) -> torch.Tensor:
        match request.model_type:
            case ModelType.DD:
                return self.inference.generate_inpainting(
                    callback=callback,
                    scheduler=self.create_scheduler(request.kwargs.get("scheduler_type")),
                    sampler=self.create_sampler(request.kwargs.get("sampler_type")),
                    **request.kwargs
                )

            case _:
                raise ValueError("Unexpected ModelType in handle_inpainting")

    def handle_extension(self, request: Request, callback: Callable) -> torch.Tensor:
        match request.model_type:
            case ModelType.DD:
                return self.inference.generate_extension(
                    callback=callback,
                    scheduler=self.create_scheduler(request.kwargs.get("scheduler_type")),
                    sampler=self.create_sampler(request.kwargs.get("sampler_type")),
                    **request.kwargs
                )
                
            case _:
                raise ValueError("Unexpected ModelType in handle_extension")
            

    def create_scheduler(self, scheduler_type: SchedulerType) -> SchedulerBase:
        match scheduler_type:
            case SchedulerType.LinearSchedule:
                return LinearSchedule(self.device_accelerator)
            
            case SchedulerType.DDPMSchedule:
                return DDPMSchedule(self.device_accelerator)
            
            case SchedulerType.SplicedDDPMCosineSchedule:
                return SplicedDDPMCosineSchedule(self.device_accelerator)
            
            case SchedulerType.LogSchedule:
                return LogSchedule(self.device_accelerator)
            
            case SchedulerType.CrashSchedule:
                return CrashSchedule(self.device_accelerator)
            

    def create_sampler(self, sampler_type: SamplerType) -> SamplerBase:
        match sampler_type:
            case SamplerType.DDPM:
                return DDPM()
            
            case SamplerType.DDIM:
                return DDIM()
            
            case SamplerType.IPLMS:
                return IPLMS()
