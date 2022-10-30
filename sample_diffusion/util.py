import os, argparse
import torchaudio
import torch
import json
import time
import math

from sample_diffusion.platform import get_torch_device_type
from dsp.post_process import post_process_audio
from dance_diffusion.model import DanceDiffusionModel
from diffusion_api.inference import Inference
from diffusion_api.util import set_seed
from diffusion_api.schedule import CrashSchedule
from diffusion_api.sampler import ImprovedPseudoLinearMultiStep

def load_audio(device, audio_path: str, sample_rate):
    if not os.path.exists(audio_path):
        raise RuntimeError(f"Audio file not found: {audio_path}")

    audio, file_sample_rate = torchaudio.load(audio_path)

    if file_sample_rate != sample_rate:
        resample = torchaudio.transforms.Resample(file_sample_rate, sample_rate)
        audio = resample(audio)

    return audio.to(device)

def save_audio(audio_out, args, seed, batch):
    output_path = get_output_folder(args, seed, batch)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    write_metadata(args, seed, batch, os.path.join(output_path, "meta.json"))
    
    for ix, sample in enumerate(audio_out):
        output_file = os.path.join(output_path, f"sample_{ix + 1}.wav")
        open(output_file, "a").close()
        
        output = sample.cpu()

        torchaudio.save(output_file, output, args.sr)

    if args.batches > 1:
        print(f"Finished batch {batch + 1} of {args.batches}.")

    # open the request_path folder in a cross-platform way
    if args.open:
        if os.name == "nt":
            os.startfile(output_path)
        elif os.name == "posix":
            os.system(f"open {output_path}")
    else:
        print(f"\nYour samples are waiting for you here: {output_path}")



def write_metadata(args, seed, batch, path):
    metadata = {
        "args": vars(args),
        "seed": seed,
        "batch": batch
    }

    write_to_json(metadata, path)


def write_to_json(obj, path):
    with open(path, "w") as f:
        obj_json = json.dumps(obj, indent=2)
        f.write(obj_json)


def get_output_folder(args, seed, batch):
    if args.input is None:
        return os.path.join(
           args.out_path, f"generations", f"{seed}_{args.steps}"
        )
    
    if len(args.input) == 1:
        return os.path.join(
            args.out_path, f"variations", f"{seed}_{args.steps}_{args.noise_level}"
        )

    file_1 = os.path.splitext(os.path.basename(args.input[0]))[0]
    file_2 = os.path.splitext(os.path.basename(args.input[1]))[0]

    return os.path.join(
        args.out_path, f"interpolations", f"{seed}_{file_1}_to_{file_2}"
    )
