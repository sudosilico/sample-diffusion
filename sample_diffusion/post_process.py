import torchaudio
import torch

def post_process_audio(audio_out, sample_rate: int, remove_dc_offset: bool = True, normalize: bool = True):
    if remove_dc_offset:
        print("Filtering DC offset...")
        audio_out = remove_dc_offset(audio_out, sample_rate)

    if normalize:
        print("Normalizing...")
        audio_out = normalize_audio(audio_out)

    return audio_out


def remove_dc_offset(audio_out, sample_rate):
    return torchaudio.functional.highpass_biquad(audio_out, sample_rate, 15, 0.707)


def normalize_audio(audio_out):
    return audio_out / torch.max(torch.abs(audio_out))
