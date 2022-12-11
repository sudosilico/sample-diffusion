import math
import torch

from typing import Callable

from dance_diffusion.base.model import ModelWrapperBase
from dance_diffusion.base.inference import InferenceBase

from diffusion_library.sampler import SamplerBase
from diffusion_library.scheduler import SchedulerBase
from diffusion_library.util import t_to_alpha_sigma

from util.util import tensor_slerp_2D


class DDInference(InferenceBase):
    def __init__(
        self,
        device_accelerator: torch.device = None,
        device_offload: torch.device = None,
        optimize_memory_use: bool = False,
        use_autocast: bool = True,
        model: ModelWrapperBase = None
    ):
        super().__init__(device_accelerator, device_offload, optimize_memory_use, use_autocast, model)


    def generate(
        self,
        callback: Callable = None,
        batch_size: int = None,
        seed: int = None,
        steps: int = None,
        scheduler: SchedulerBase = None,
        scheduler_args = None,
        sampler: SamplerBase = None,
        sampler_args = None,
        **kwargs
    ) -> torch.Tensor:
        self.generator.manual_seed(seed)
        
        ts = scheduler.create(steps, scheduler_args=scheduler_args)
        x_T = torch.randn([batch_size, 2, self.model.chunk_size], generator=self.generator, device=self.device_accelerator)
        
        with self.offload_context(self.model.model):
            return sampler.sample(
                x_T,
                ts,
                self.model.model,
                callback,
                sampler_args=sampler_args
            ).float()


    def generate_variation(
        self,
        callback: Callable = None,
        batch_size: int = None,
        seed: int = None,
        audio_source: torch.Tensor = None,
        noise_level: float = None,
        steps: int = None,
        scheduler: SchedulerBase = None,
        scheduler_args = None,
        sampler: SamplerBase = None,
        sampler_args = None,
        **kwargs
    ) -> torch.Tensor:
        self.generator.manual_seed(seed)
        
        ts = scheduler.create(steps, noise_level, scheduler_args=scheduler_args)
        alpha_T, sigma_T = t_to_alpha_sigma(ts[0])
        
        audio_source = audio_source[None, :, :].expand(batch_size, -1, -1)
        x_T = alpha_T * audio_source + sigma_T * torch.randn(audio_source.shape, device=audio_source.device, generator=self.generator)
        with self.offload_context(self.model.model):
            return sampler.sample(
                x_T,
                ts,
                self.model.model,
                callback,
                sampler_args=sampler_args
            ).float()


    def generate_interpolation(
        self,
        callback: Callable = None,
        batch_size: int = None,
        # seed: int = None,
        interpolation_positions = None, # list[float] or torch.Tensor(1D)
        audio_source: torch.Tensor = None,
        audio_target: torch.Tensor = None,
        noise_level: float = None,
        steps: int = None,
        scheduler: SchedulerBase = None,
        scheduler_args = None,
        sampler: SamplerBase = None,
        sampler_args = None,
        **kwargs
    ) -> torch.Tensor:
        # self.generator.manual_seed(seed)
        
        ts = scheduler.create(steps, noise_level, scheduler_args=scheduler_args)
        x_0 = torch.cat([audio_source[None, :, :], audio_target[None, :, :]], dim=0)
        
        with self.offload_context(self.model.model):
            x_T = sampler.sample(
                x_0,
                ts.flip(0),
                self.model.model,
                callback,
                sampler_args=sampler_args
            )

        audio_output = torch.empty([0, 2, x_T.shape[2]], device=self.device_accelerator)
        #audio_output = torch.cat([audio_output,x_0[0][None, :, :]], dim=0)
        
        total_samples = len(interpolation_positions)
        batches = math.ceil(total_samples / batch_size)

        for batch in range(batches):
            n_samples = min(batch_size, total_samples)
            total_samples = total_samples - n_samples
            
            buffer = torch.empty([n_samples, 2, x_T.shape[2]], device=self.device_accelerator)
            
            for sample in range(n_samples):
                index = batch * batch_size + sample
                buffer[sample,:,:] = tensor_slerp_2D(x_T[0], x_T[1], interpolation_positions[index])[None, :, :]
                
            with self.offload_context(self.model.model):
                
                audio_output = torch.cat(
                    [audio_output,
                    sampler.sample(
                        buffer,
                        ts,
                        self.model.model,
                        callback,
                        sampler_args=sampler_args
                    )],
                    dim=0
                )

        #audio_output = torch.cat([audio_output,x_0[1][None, :, :]], dim=0)
        
        return audio_output.float()


    def generate_inpainting(
        self,
        callback: Callable = None,
        batch_size: int = None,
        seed: int = None,
        audio_source: torch.Tensor = None,
        mask: torch.Tensor = None,
        steps: int = None,
        scheduler: SchedulerBase = None,
        scheduler_args = None,
        sampler: SamplerBase = None,
        sampler_args = None,
        resamples: int = None,
        **kwargs
    ) -> torch.Tensor:
        self.generator.manual_seed(seed)
        
        ts = scheduler.create(steps, scheduler_args=scheduler_args)
        audio_source = audio_source[None, :, :].expand(batch_size, -1, -1)
        
        with self.offload_context(self.model.model):
            return sampler.inpaint(
                audio_source,
                mask,
                ts,
                resamples,
                self.model.model,
                callback,
                sampler_args=sampler_args
            ).float()


    def generate_extension(
        self,
        callback: Callable = None,
        batch_size: int = None,
        seed: int = None,
        audio_source: torch.Tensor = None,
        steps: int = None,
        scheduler: SchedulerBase = None,
        scheduler_args = None,
        sampler: SamplerBase = None,
        sampler_args = None,
        resamples: int = None,
        keep_start: bool = None,
        **kwargs
    ) -> torch.Tensor:
        
        half_chunk_size = self.model.chunk_size // 2
        chunk = torch.cat([audio_source[:, -half_chunk_size:], torch.randn([2, half_chunk_size], generator=self.generator, device=self.device_accelerator)], dim=1)
        
        mask = torch.cat(
            [torch.ones([batch_size, 2, half_chunk_size], dtype=torch.bool, device=self.device_accelerator),
            torch.zeros([batch_size, 2, half_chunk_size], dtype=torch.bool, device=self.device_accelerator)],
            dim=2 
        )
        
        output = self.generate_inpainting(
            callback,
            batch_size,
            seed,
            chunk,
            mask,
            steps,
            scheduler,
            scheduler_args,
            sampler,
            sampler_args,
            resamples
        )
        
        if (keep_start):
            return torch.cat(
                [audio_source[None, :, :].expand(batch_size, -1, -1),
                 output[:, :, -half_chunk_size:]],
                dim=2
            )
        else:
            return output[:, :, -half_chunk_size:]
        