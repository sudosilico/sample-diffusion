import gc
from lib2to3.pgen2.tokenize import TokenError
import torch
from dance_diffusion.model import ModelBase
from diffusion_api.sampler import SamplerBase
from diffusion_api.schedule import ScheduleBase
from diffusion_api.util import t_to_alpha_sigma, tensor_slerp_2D


class Inference:
    def __init__(
        self,
        device: torch.device = None,
        model: ModelBase = None,
        sampler: SamplerBase = None,
        scheduler: ScheduleBase = None,
    ):
        self.device = device
        self.model = model
        self.sampler = sampler
        self.scheduler = scheduler

    def generate_unconditional(
        self,
        batch_size: int = 1,
        steps: int = 25,
        callback=None
    ):

        torch.cuda.empty_cache()
        gc.collect()

        x_T = torch.randn([batch_size, 2, self.model.chunk_size], device=self.device)
        ts = self.scheduler.create(steps)

        return self.sampler.sample(x_T, ts, callback)

    def generate_variation(
        self,
        batch_size: int = 1,
        steps: int = 25,
        audio_input: torch.Tensor = None,
        noise_level: float = 0.7,
        callback=None,
    ):
        """
        Invariants:
            `audio_input` must be of a length that is a multiple of the model chunk_size for proper behavior.
            `audio_input` must be on the same device as self.device
        """

        torch.cuda.empty_cache()
        gc.collect()

        ts = self.scheduler.create(steps, noise_level)
        alpha_T, sigma_T = t_to_alpha_sigma(t=ts[0])

        audio_input = audio_input[None, :, :].expand(batch_size, -1, -1)
        noise = torch.randn_like(audio_input)
        x_T = alpha_T * audio_input + sigma_T * noise

        return self.sampler.sample(x_T, ts, callback)

    def generate_interpolation(
        self,
        samples: int,
        steps: int,
        audio_source: torch.Tensor,
        audio_target: torch.Tensor,
        noise_level: float = 1.0,
        callback=None,
    ):
        """
        Invariants:
            `audio_source` and `audio_target` must be of a length that is a multiple of the model chunk_size for proper behavior.
        """
        torch.cuda.empty_cache()
        gc.collect()

        ts = self.scheduler.create(steps, noise_level)

        x_0 = torch.cat([audio_source[None, :, :], audio_target[None, :, :]], dim=0)
        x_T = self.sampler.sample(x_0, ts.flip(0), callback)

        audio_output = []
        audio_output.append(
            x_0[0]
        )  # audio_output.append(self.sampler.sample(x_T[0][None, :, :], ts).squeeze(dim=0))

        for sample in range(samples - 2):
            audio_output.append(
                self.sampler.sample(
                    tensor_slerp_2D(x_T[0], x_T[1], (1 + sample) / (samples - 1))[
                        None, :, :
                    ],
                    ts,
                ).squeeze(dim=0)
            )

        audio_output.append(
            x_0[1]
        )  # audio_output.append(self.sampler.sample(x_T[1][None, :, :], ts).squeeze(dim=0))

        return audio_output


    def generate_extension(
        self,
        batch_size: int,
        steps: int,
        resamples: int,
        extensions: int, 
        audio_input: torch.Tensor,
        callback=None
    ):
        torch.cuda.empty_cache()
        gc.collect()
        
        ts = self.scheduler.create(steps)
        
        audio_input = audio_input[None, :, :].expand(batch_size, -1, -1)
        
        chunk_size = self.model.chunk_size
        half_chunk_size = chunk_size // 2
        audio_input_size = audio_input.size(2)
        total_size = audio_input_size + (half_chunk_size * extensions)
        extension_size = (extensions + 1) * half_chunk_size
        
        noise_tensor = torch.randn([batch_size, 2, extension_size], device=self.device)
    
        mask = torch.cat(
            [torch.ones([batch_size, 2, half_chunk_size], dtype=torch.bool, device=self.device),
            torch.zeros([batch_size, 2, half_chunk_size], dtype=torch.bool, device=self.device)],
            dim=2
        )

        last_half_chunk = audio_input[:, :, -half_chunk_size:]
        
        audio_output = torch.empty([batch_size, 2, total_size], device=self.device)
        audio_output[:, :, 0:audio_input_size] = audio_input
        
        for chunk in range(extensions):
            next_half_chunk = noise_tensor[:, :, (chunk + 1) * half_chunk_size:(chunk + 2) * half_chunk_size]
            current_chunk = torch.cat([last_half_chunk, next_half_chunk], dim=2)
            current_chunk = self.sampler.inpaint(current_chunk, mask, ts, resamples, callback)
            last_half_chunk = current_chunk[:, :, -half_chunk_size:]
            audio_output[:, :, (audio_input_size + chunk * half_chunk_size):(audio_input_size + (chunk + 1) * half_chunk_size)] = last_half_chunk
    
        return audio_output
