import torch
from torch import nn
from audio_diffusion.models import DiffusionAttnUnet1D

def load_model(model_args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_ph = instantiate_model(model_args.spc, model_args.sr)
    model = load_state_from_checkpoint(device, model_ph, model_args.ckpt)

    return model, device

def instantiate_model(chunk_size, model_sample_rate):
    class DiffusionInference(nn.Module):
        def __init__(self, global_args):
            super().__init__()

            self.diffusion_ema = DiffusionAttnUnet1D(global_args, n_attn_layers = 4)
            self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

    class Object(object):
        pass

    model_args = Object()
    model_args.sample_size = chunk_size
    model_args.sample_rate = model_sample_rate
    model_args.latent_dim = 0

    return DiffusionInference(model_args)

def load_state_from_checkpoint(device, model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"],strict = False)
    return model.requires_grad_(False).to(device)

