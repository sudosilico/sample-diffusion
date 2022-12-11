import os
import torch
import torchaudio

def tensor_slerp_2D(a: torch.Tensor, b: torch.Tensor, t: float):
    slerped = torch.empty_like(a)
    
    for channel in range(a.size(0)):
        slerped[channel] = tensor_slerp(a[channel], b[channel], t)
    
    return slerped


def tensor_slerp(a: torch.Tensor, b: torch.Tensor, t: float):
    omega = torch.arccos(torch.dot(a / torch.linalg.norm(a), b / torch.linalg.norm(b)))
    so = torch.sin(omega)
    return torch.sin((1.0-t) * omega) / so * a + torch.sin(t * omega) / so * b


def load_audio(device, audio_path: str, sample_rate):
    
    if not os.path.exists(audio_path):
        raise RuntimeError(f"Audio file not found: {audio_path}")

    audio, file_sample_rate = torchaudio.load(audio_path)

    if file_sample_rate != sample_rate:
        resample = torchaudio.transforms.Resample(file_sample_rate, sample_rate)
        audio = resample(audio)

    return audio.to(device)


def save_audio(audio_out, output_path: str, sample_rate, id_str:str = None):

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for ix, sample in enumerate(audio_out):
        output_file = os.path.join(output_path, f"sample_{id_str}_{ix + 1}.wav" if(id_str!=None) else f"sample_{ix + 1}.wav")
        open(output_file, "a").close()
        
        output = sample.cpu()

        torchaudio.save(output_file, output, sample_rate)


def cropper(samples: int, randomize: bool = True):

    def crop(source: torch.Tensor, offset_in: int = None) -> torch.Tensor:
        n_channels, n_samples = source.shape
        
        offset = 0
        if (offset_in):
            offset = min(offset_in, n_samples - samples)
        elif (randomize):
            offset = torch.randint(0, max(0, n_samples - samples) + 1, []).item()
        
        chunk = source.new_zeros([n_channels, samples])
        chunk [:, :min(n_samples, samples)] = source[:, offset:offset + samples]
        
        return chunk

    return crop