import gc
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
        self, batch_size: int = 1, steps: int = 25, callback=None
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
