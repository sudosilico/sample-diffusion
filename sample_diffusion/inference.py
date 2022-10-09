from contextlib import nullcontext
import math
import gc
import torch
from audio_diffusion.utils import Stereo, PadCrop
from tqdm.auto import trange
from sample_diffusion.platform import get_torch_device_type

from diffusion import utils
import sys

def make_eps_model_fn(model):
    def eps_model_fn(x, t, **extra_args):
        alphas, sigmas = utils.t_to_alpha_sigma(t)
        v = model(x, t, **extra_args)
        eps = x * utils.append_dims(sigmas, x.ndim) + v * utils.append_dims(alphas, x.ndim)
        return eps
    return eps_model_fn


def make_autocast_model_fn(model):
    def autocast_model_fn(*args, **kwargs):
        device = get_torch_device_type()
        precision_scope = torch.autocast
        if device in ['mps', 'cpu']:
            precision_scope = nullcontext
        with precision_scope(device):
            m = model(*args, **kwargs).float() 
            return m
    return autocast_model_fn


def transfer(x, eps, t_1, t_2):
    alphas, sigmas = utils.t_to_alpha_sigma(t_1)
    next_alphas, next_sigmas = utils.t_to_alpha_sigma(t_2)

    # Fix for apple silicon mps
    # --
    # On mps, `alphas` (thus, v1) is zero. `pred` divides by zero, resulting in NaNs.
    # This is a hacky workaround to avoid the division by zero, there should be a more appropriate fix.
    v0 = utils.append_dims(sigmas, x.ndim)
    v1 = utils.append_dims(alphas, x.ndim)
    nonzero = torch.tensor(sys.float_info.epsilon).to(v1.device)
    pred = (x - eps * v0) / torch.where(v1 == 0, nonzero, v1)

    x = pred * utils.append_dims(next_alphas, x.ndim) + eps * utils.append_dims(next_sigmas, x.ndim)
    return x, pred

def iplms_step(model, x, old_eps, t_1, t_2, extra_args):
    eps_model_fn = make_eps_model_fn(model)
    eps = eps_model_fn(x, t_1, **extra_args)
    if len(old_eps) == 0:
        eps_prime = eps
    elif len(old_eps) == 1:
        eps_prime = (3/2 * eps - 1/2 * old_eps[-1])
    elif len(old_eps) == 2:
        eps_prime = (23/12 * eps - 16/12 * old_eps[-1] + 5/12 * old_eps[-2])
    else:
        eps_prime = (55/24 * eps - 59/24 * old_eps[-1] + 37/24 * old_eps[-2] - 9/24 * old_eps[-3])
    x_new, _ = transfer(x, eps_prime, t_1, t_2)
    _, pred = transfer(x, eps, t_1, t_2)

    return x_new, eps, pred


@torch.no_grad()
def iplms_sample(model, x, steps, extra_args, is_reverse=False, callback=None):
    """Draws samples from a model given starting noise using fourth order
    Improved Pseudo Linear Multistep."""
    ts = x.new_ones([x.shape[0]])
    model_fn = make_autocast_model_fn(model)
    if not is_reverse:
        steps = torch.cat([steps, steps.new_zeros([1])])
    old_eps = []
    for i in trange(len(steps) - 1, disable=None):
        x, eps, pred = iplms_step(model_fn, x, old_eps, steps[i] * ts, steps[i + 1] * ts, extra_args)
        if len(old_eps) >= 3:
            old_eps.pop(0)
        old_eps.append(eps)
        if callback is not None:
            callback({'x': x, 'i': i, 't': steps[i], 'pred': pred})
    return x


def rand2audio(model, batch_size: int = 1, n_steps: int = 25, callback=None):
    torch.cuda.empty_cache()
    gc.collect()

    noise = torch.randn([batch_size, 2, model.chunk_size]).to(model.device, non_blocking=False)
    t = torch.linspace(1, 0, n_steps + 1, device=model.device)[:-1]
    step_list = get_crash_schedule(t)

    audio_out = iplms_sample(
        model.diffusion, noise, step_list, {}, callback=callback
    )

    return audio_out


def audio2audio(
    model,
    batch_size,
    n_steps,
    audio_input,
    noise_level,
    length_multiplier,
    callback=None,
):
    print("Length multiplier: ", length_multiplier)
    effective_length = model.chunk_size * length_multiplier

    torch.cuda.empty_cache()
    gc.collect()

    augs = torch.nn.Sequential(PadCrop(effective_length, randomize=True), Stereo())
    audio = augs(audio_input).unsqueeze(0).repeat([batch_size, 1, 1])

    t = torch.linspace(0, 1, n_steps + 1, device=model.device)
    step_list = get_crash_schedule(t)
    step_list = step_list[step_list < noise_level]

    alpha, sigma = t_to_alpha_sigma(step_list[-1])
    noise = torch.randn([batch_size, 2, effective_length], device=model.device)
    noised_audio = audio * alpha + noise * sigma

    return iplms_sample(
        model.diffusion,
        noised_audio,
        step_list.flip(0)[:-1],
        {},
        callback=callback,
    )


def get_crash_schedule(t):
    sigma = torch.sin(t * math.pi / 2) ** 2
    alpha = (1 - sigma**2) ** 0.5
    return alpha_sigma_to_t(alpha, sigma)


def alpha_sigma_to_t(alpha, sigma):
    return torch.atan2(sigma, alpha) / math.pi * 2


def t_to_alpha_sigma(t):
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)
