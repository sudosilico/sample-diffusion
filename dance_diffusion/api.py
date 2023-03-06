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


class ModelType(str, enum.Enum):
    DD = "DD"
    
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

class SchedulerType(str, enum.Enum):
    LinearSchedule = "LinearSchedule"
    DDPMSchedule = "DDPMSchedule"
    SplicedDDPMCosineSchedule = "SplicedDDPMCosineSchedule"
    LogSchedule = "LogSchedule"
    CrashSchedule = "CrashSchedule"


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
            
        handlers_by_request_type = {
            RequestType.Generation: self.handle_generation,
            RequestType.Variation: self.handle_variation,
            RequestType.Interpolation: self.handle_interpolation,
            RequestType.Inpainting: self.handle_inpainting,
            RequestType.Extension: self.handle_extension,
        }
        
        handler = handlers_by_request_type.get(request.request_type)
        
        if handler:
            return Response(handler(request, callback))
        
        raise ValueError(f"Unexpected RequestType in process_request: '{request.request_type}'")
    

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
        schedulers_by_type = {
            SchedulerType.LinearSchedule: LinearSchedule,
            SchedulerType.DDPMSchedule: DDPMSchedule,
            SchedulerType.SplicedDDPMCosineSchedule: SplicedDDPMCosineSchedule,
            SchedulerType.LogSchedule: LogSchedule,
            SchedulerType.CrashSchedule: CrashSchedule,
        }
        
        Scheduler = schedulers_by_type.get(scheduler_type)
        
        if Scheduler:
            return Scheduler(self.device_accelerator)
            
    def create_sampler(self, sampler_type: SamplerType) -> SamplerBase:
        samplers_by_type = {
            SamplerType.DDPM: DDPM,
            SamplerType.DDIM: DDIM,
            SamplerType.IPLMS: IPLMS,
        }
        
        Sampler = samplers_by_type.get(sampler_type)
        
        if Sampler:
            return Sampler()
