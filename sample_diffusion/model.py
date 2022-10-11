from math import ceil
import os
import torch
import torchaudio
from torch import nn
from audio_diffusion.models import DiffusionAttnUnet1D
from sample_diffusion.inference import audio2audio, rand2audio
from sample_diffusion.platform import get_torch_device_type
from pytorch_lightning import seed_everything


class GlobalArgs(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class DiffusionInference(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.diffusion_ema = DiffusionAttnUnet1D(GlobalArgs(**kwargs), n_attn_layers=4)
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)


class Model:
    _module: DiffusionInference = None
    _device_type: str = None
    device: torch.device = None
    model_path: str = None
    chunk_size: int = 65536
    sample_rate: int = 48000

    def __init__(
        self,
        force_cpu: bool = False,
    ):
        if force_cpu:
            self._device_type = "cpu"
        else:
            self._device_type = get_torch_device_type()

        self.device = torch.device(self._device_type)

    def load(self, model_path, chunk_size=65536, sample_rate=48000):
        if self.model_path == model_path:
            return

        self._module = DiffusionInference(
            sample_size=chunk_size,
            sample_rate=sample_rate,
            latent_dim=0,
        )

        self.model_path = model_path
        
        self._module.load_state_dict(
            torch.load(model_path, map_location=self.device)["state_dict"], strict=False
        )

        self._module = self._module.requires_grad_(False).to(self.device)

    @property
    def diffusion(self):
        if self._module is None:
            raise RuntimeError("Model not loaded.")

        return self._module.diffusion_ema

    def generate(
        self, seed: int = -1, samples: int = 1, steps: int = 25, callback=None
    ):
        """Generates new unconditional audio samples."""

        seed = self.seed(seed)
        audio_out = rand2audio(self, samples, steps, callback)

        return audio_out, seed

    def process_audio_file(
        self,
        audio_path: str,
        noise_level: float = 0.7,
        length_multiplier: int = -1,
        seed: int = -1,
        samples: int = 1,
        steps: int = 25,
        callback=None,
    ):
        """Generate new audio samples from a given audio file path."""

        audio_in = self._load_audio_file(audio_path)

        # The sample length, in multiples of self.chunk_size. Will use the largest that fits all of the audio if not specified or -1 used.
        length = (
            length_multiplier
            if length_multiplier != -1
            else (ceil(audio_in.shape[-1] / float(self.chunk_size)))
        )

        seed = self.seed(seed)
        audio_out = audio2audio(
            self, samples, steps, audio_in, noise_level, length, callback
        )

        return audio_out, seed

    def _load_audio_file(self, audio_path: str):
        if self.device is None:
            raise RuntimeError("Model not loaded.")

        if not os.path.exists(audio_path):
            raise RuntimeError(f"Audio file not found: {audio_path}")

        audio, file_sample_rate = torchaudio.load(audio_path)

        # Resample the audio to the model's sample rate if necessary.
        if file_sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                file_sample_rate, self.sample_rate
            )
            audio = resampler(audio)

        return audio.to(self.device)

    def seed(self, seed):
        """
            Sets the preferred seed for the model, returning the seed that will be used. 
            If -1 is passed in, a random seed will be used.
            If the passed seed is too large, it will be modulo'd by the maximum seed value.
        """

        seed = seed if seed != -1 else torch.seed()
        seed = seed % 4294967295
        seed_everything(seed)
        self.current_seed = seed
        return seed
