import torch
from torch import nn
from audio_diffusion.models import DiffusionAttnUnet1D

from sample_diffusion.platform import get_torch_device_type

# def instantiate_model(chunk_size: int, sample_rate: int) -> DiffusionInference:
#     return DiffusionInference(
#         sample_size=chunk_size,
#         sample_Rate=sample_rate,
#         latent_dim=0,
#     )

# def load_state_from_checkpoint(device, model, checkpoint_path):
#     model.load_state_dict(torch.load(checkpoint_path, map_location=device)["state_dict"], strict=False)
#     return model.requires_grad_(False).to(device)

class Model:
    class DiffusionInference(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

            self.diffusion_ema = DiffusionAttnUnet1D(**kwargs, n_attn_layers=4)
            self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
    
    _module: DiffusionInference = None
    _device_type: str = None
    device: torch.device = None
    model_path: str = None
    chunk_size: int = 65536
    sample_rate: int = 48000

    def __init__(self, chunk_size:int = 65535, sample_rate:int = 48000, device_type:str=None):
        print("init model")
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self._device_type = device_type

        if self._device_type is None:
            self._device_type = get_torch_device_type()

        print("device type", self._device_type)

        print("init model 2")
        self.device = torch.device(self._device_type)

        print("init model 3")
        self._module = Model.DiffusionInference(
            sample_size=self.chunk_size,
            sample_rate=self.sample_rate,
            latent_dim=0,
        )

        print("init model 4")

    def load(self, model_path, chunk_size=65536, sample_rate=48000):
        if self.model_path == model_path:
            return

        if (self.chunk_size != chunk_size) or (self.sample_rate != sample_rate):
            self._module = Model.DiffusionInference(
                sample_size=chunk_size,
                sample_rate=sample_rate,
                latent_dim=0,
            )

        self.model_path = model_path
        self._module.load_state_dict(torch.load(model_path, map_location=self.device)["state_dict"], strict=False)
        self._module = self._module.requires_grad_(False).to(self.device)

    @property
    def diffusion_ema(self):
        return self._module.diffusion_ema

# class ModelInfo:
#     def __init__(self, model, device, chunk_size):
#         self.model = model
#         self.device = device
#         self.chunk_size = chunk_size

#     def switch_models(self, ckpt="models/model.ckpt", sample_rate=48000, chunk_size=65536):
#         device_type = get_torch_device_type()
#         device = torch.device(device_type)

#         model_ph = instantiate_model(chunk_size, sample_rate)
#         model = load_state_from_checkpoint(device, model_ph, ckpt)

#         self.model = model
#         self.device = device
#         self.chunk_size = chunk_size
        
# def get_torch_device_type():
#     if is_mps_available():
#         return "mps"

#     if torch.cuda.is_available():
#         return "cuda"

#     return "cpu"

# def is_mps_available():
#     try:
#         return torch.backends.mps.is_available()
#     except:
#         return False


# def load_model(model_args):
#     chunk_size = model_args.spc
#     sample_rate = model_args.sr
#     ckpt = model_args.ckpt

#     device_type = get_torch_device_type()
#     device = torch.device(device_type)

#     model_ph = instantiate_model(chunk_size, sample_rate)
#     model = load_state_from_checkpoint(device, model_ph, ckpt)

#     return ModelInfo(model, device, chunk_size)


# def load_state_from_checkpoint(device, model, checkpoint_path):
#     model.load_state_dict(torch.load(checkpoint_path, map_location=device)["state_dict"], strict=False)
#     return model.requires_grad_(False).to(device)
