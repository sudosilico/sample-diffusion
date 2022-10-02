import torch
from torch import nn
from audio_diffusion.models import DiffusionAttnUnet1D


class ModelInfo:
    def __init__(self, model, device, chunk_size):
        self.model = model
        self.device = device
        self.chunk_size = chunk_size

    def switch_models(self, ckpt="models/model.ckpt", sample_rate=48000, chunk_size=65536):
        print("Switching models...")
        print("Unloading previous model...")

        del self.model
        del device

        print("Emptying CUDA cache...")

        torch.cuda.empty_cache()

        print("Loading new model...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_ph = instantiate_model(chunk_size, sample_rate)
        model = load_state_from_checkpoint(device, model_ph, ckpt)

        self.model = model
        self.device = device
        self.chunk_size = chunk_size

        print(f"Swapped to model: '{ckpt}'")
        
        

def load_model(model_args):
    chunk_size = model_args.spc
    sample_rate = model_args.sr
    ckpt = model_args.ckpt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_ph = instantiate_model(chunk_size, sample_rate)
    model = load_state_from_checkpoint(device, model_ph, ckpt)

    return ModelInfo(model, device, chunk_size)


def instantiate_model(chunk_size, model_sample_rate):
    class DiffusionInference(nn.Module):
        def __init__(self, global_args):
            super().__init__()

            self.diffusion_ema = DiffusionAttnUnet1D(global_args, n_attn_layers=4)
            self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

    class Object(object):
        pass

    model_args = Object()
    model_args.sample_size = chunk_size
    model_args.sample_rate = model_sample_rate
    model_args.latent_dim = 0

    return DiffusionInference(model_args)


def load_state_from_checkpoint(device, model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"], strict=False)
    return model.requires_grad_(False).to(device)
