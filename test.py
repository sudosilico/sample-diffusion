import os
import torch
import torchaudio
import time
import math

from sample_diffusion.platform import get_torch_device_type
from dance_diffusion.model import DanceDiffusionModel
from diffusion_api.inference import Inference
from diffusion_api.util import set_seed
from diffusion_api.schedule import CrashSchedule
from diffusion_api.sampler import DenoisingDiffusionProbabilisticModel, ImprovedPseudoLinearMultiStep
from sample_diffusion.util import load_audio

def save_audio(audio_out):
    output_path = "audio_out/test/"

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for ix, sample in enumerate(audio_out):
        output_file = os.path.join(output_path, f"sample_{ix + 1}.wav")
        open(output_file, "a").close()
        
        output = sample.cpu()

        torchaudio.save(output_file, output, 48000)

def call_me(args):
    save_audio(args.get('pred'))

def main():
    
    device_type = get_torch_device_type()
    print("Using device:", device_type)
    device = torch.device(device_type)
    
    model = DanceDiffusionModel(device)
    model.load(r"C:\Users\mrhya\Documents\models\jmann.ckpt", 65536, 48000)
    
    scheduler = CrashSchedule(device)
    sampler = DenoisingDiffusionProbabilisticModel(model)
    inference = Inference(device, model, sampler, scheduler)
    
    audio_input = load_audio(inference.device, r"C:\Users\mrhya\Documents\Git\Projects\SDA\sample-diffusion\audio_out\test2.wav", 48000)
    
    seed = set_seed(42)
    save_audio(inference.generate_extension(8, 100, 4, 1, audio_input, call_me))
    
    
if __name__ == "__main__":
    main()