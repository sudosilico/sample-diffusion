import math
import torch

from pytorch_lightning import seed_everything


def pad_dims(a,b):
    return a[(...,) + (None,) * (b.ndim - a.ndim)]


def t_to_alpha_sigma(t):
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


def set_seed(seed):
    """
        Sets the preferred seed for the model, returning the seed that will be used. 
        If -1 is passed in, a random seed will be used.
        If the passed seed is too large, it will be modulo'd by the maximum seed value.
    """

    seed = seed if seed != -1 else torch.seed()
    seed = seed % 4294967295
    seed_everything(seed)
    return seed


def tensor_slerp_2D(a: torch.Tensor, b: torch.Tensor, t: float):
    slerped = torch.empty_like(a)
    
    for channel in range(a.size(0)):
        slerped[channel] = tensor_slerp(a[channel], b[channel], t)
    
    return slerped

def tensor_slerp(a: torch.Tensor, b: torch.Tensor, t: float):
    """Spherical linear interpolation between two tensors."""
    omega = torch.arccos(torch.dot(a/torch.linalg.norm(a), b/torch.linalg.norm(b)))
    so = torch.sin(omega)
    return torch.sin((1.0-t)*omega) / so * a + torch.sin(t*omega)/so * b
