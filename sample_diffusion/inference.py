import math
import gc
import torchaudio
import torch
from diffusion import sampling
from audio_diffusion.utils import Stereo, PadCrop
from pytorch_lightning import seed_everything


def set_seed(new_seed):
    if new_seed != -1:
        seed = new_seed
    else:
        seed = torch.seed()
    seed = seed % 4294967295
    seed_everything(seed)
    return seed


def generate_audio(seed, samples, steps, model_info, callback=None):
    seed = set_seed(seed)

    audio_out = rand2audio(model_info, samples, steps, callback)

    return audio_out, seed


def process_audio(
    input_path,
    sample_rate,
    sample_length_multiplier,
    noise_level,
    seed,
    samples,
    steps,
    model_info,
    callback=None,
):
    seed = set_seed(seed)

    audio_in = load_to_device(model_info.device, input_path, sample_rate)

    audio_out = audio2audio(
        model_info,
        samples,
        steps,
        audio_in,
        noise_level,
        sample_length_multiplier,
        callback,
    )

    return audio_out, seed


def rand2audio(model_info, batch_size, n_steps, callback=None):
    torch.cuda.empty_cache()
    gc.collect()

    noise = torch.randn([batch_size, 2, model_info.chunk_size]).to(model_info.device)
    t = torch.linspace(1, 0, n_steps + 1, device=model_info.device)[:-1]
    step_list = get_crash_schedule(t)

    return sampling.iplms_sample(
        model_info.model.diffusion_ema, noise, step_list, {}, callback=callback
    ).clamp(-1, 1)


def audio2audio(
    model_info,
    batch_size,
    n_steps,
    audio_input,
    noise_level,
    sample_length_multiplier,
    callback=None,
):
    effective_length = model_info.chunk_size * sample_length_multiplier

    torch.cuda.empty_cache()
    gc.collect()

    augs = torch.nn.Sequential(PadCrop(effective_length, randomize=True), Stereo())
    audio = augs(audio_input).unsqueeze(0).repeat([batch_size, 1, 1])

    t = torch.linspace(0, 1, n_steps + 1, device=model_info.device)
    step_list = get_crash_schedule(t)
    step_list = step_list[step_list < noise_level]

    alpha, sigma = t_to_alpha_sigma(step_list[-1])
    noise = torch.randn([batch_size, 2, effective_length], device=model_info.device)
    noised_audio = audio * alpha + noise * sigma

    return sampling.iplms_sample(
        model_info.model.diffusion_ema,
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


def load_to_device(device, path, sr):
    audio, file_sr = torchaudio.load(path)
    if (sr is not None) and (sr != file_sr):
        print(
            f"Resampling from {file_sr} (file sample rate) to {sr} (model sample rate)."
        )
        audio = torchaudio.transforms.Resample(file_sr, sr)(audio)
    return audio.to(device)
