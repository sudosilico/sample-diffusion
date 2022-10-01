import math, gc
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


def generate_audio(generation_args, seed, args, device, model, callback=None):
    noise_level = generation_args.noise_level
    length = generation_args.sample_length_multiplier
    input_path = generation_args.input
    input_sr = generation_args.input_sr

    seed = set_seed(seed)

    spc = args.spc
    n_samples = args.n_samples
    n_steps = args.n_steps

    if generation_args.input:
        audio_in = load_to_device(device, input_path, input_sr)
        audio_out = audio2audio(
            device,
            model.diffusion_ema,
            spc,
            n_samples,
            n_steps,
            audio_in,
            noise_level,
            length,
            callback,
        )
    else:
        audio_out = rand2audio(
            device, model.diffusion_ema, spc, n_samples, n_steps, callback
        )

    return audio_out, seed


def process_audio(args, device, model, callback=None):
    audio_out, seed = generate_audio(args, args.seed, args, device, model, callback)
    return audio_out, seed


def rand2audio(device, model, chunk_size, batch_size, n_steps, callback=None):
    torch.cuda.empty_cache()
    gc.collect()

    noise = torch.randn([batch_size, 2, chunk_size]).to(device)
    t = torch.linspace(1, 0, n_steps + 1, device=device)[:-1]
    step_list = get_crash_schedule(t)

    return sampling.iplms_sample(model, noise, step_list, {}, callback=callback).clamp(
        -1, 1
    )


def audio2audio(
    device,
    model,
    chunk_size,
    batch_size,
    n_steps,
    audio_input,
    noise_level,
    sample_length_multiplier,
    callback=None,
):
    effective_length = chunk_size * sample_length_multiplier

    torch.cuda.empty_cache()
    gc.collect()

    augs = torch.nn.Sequential(PadCrop(effective_length, randomize=True), Stereo())
    audio = augs(audio_input).unsqueeze(0).repeat([batch_size, 1, 1])

    t = torch.linspace(0, 1, n_steps + 1, device=device)
    step_list = get_crash_schedule(t)
    step_list = step_list[step_list < noise_level]

    alpha, sigma = t_to_alpha_sigma(step_list[-1])
    noise = torch.randn([batch_size, 2, effective_length], device="cuda")
    noised_audio = audio * alpha + noise * sigma

    return sampling.iplms_sample(
        model, noised_audio, step_list.flip(0)[:-1], {}, callback=callback
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
    if sr != file_sr:
        audio = torchaudio.transforms.Resample(file_sr, sr)(audio)
    return audio.to(device)
