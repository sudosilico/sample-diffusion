import os, argparse
import torchaudio
import torch
import json
import time
import math

from diffusion.util import set_seed
from diffusion.inference import Inference
from diffusion.schedule import CrashSchedule
from dance_diffusion.model import DanceDiffusionModel
from diffusion.sampler import ImprovedPseudoLinearMultiStep

from sample_diffusion.platform import get_torch_device_type
from dsp.post_process import post_process_audio


def main():
    
    args = parse_cli_args()
    
    device = torch.device(get_torch_device_type())
    
    model = DanceDiffusionModel(device)
    model.load(args.ckpt, args.spc, args.sr)
    
    scheduler = CrashSchedule(device)
    sampler = ImprovedPseudoLinearMultiStep(model)
    inference = Inference(device, model, sampler, scheduler)
    
    start_time = time.process_time()

    for batch in range(args.batches):
        audio_out, seed = perform_batch(inference, args.seed + batch, args)

        audio_out = post_process_audio(
            audio_out, 
            sample_rate=args.sr, 
            remove_dc_offset_=args.remove_dc_offset, 
            normalize=args.normalize
        )

        save_audio(audio_out, args, seed, batch)

    end_time = time.process_time()
    elapsed = end_time - start_time

    print(
        f"Done! Generated {args.samples * args.batches} samples in {elapsed} seconds."
    )


def perform_batch(inference: Inference, seed: int, args):
    
    seed = set_seed(seed)
    
    if args.input is None:
        
        return inference.generate_unconditional(
            args.samples,
            args.steps
        ), seed

    if len(args.input) == 1:
        
        audio_input = load_audio(inference.device, args.input[0], args.sr)
        
        audio_input_size = audio_input.size(dim=1)
        min_length_multiplier = math.ceil(audio_input_size / args.spc)
        if (args.length_multiplier == -1):
            audio_input = torch.nn.functional.pad(audio_input, (0, args.spc * max([min_length_multiplier, args.length_multiplier]) - audio_input_size), "constant", 0)
        else:
            audio_input = audio_input[:,:args.spc * args.length_multiplier]
        
        return inference.generate_variation(
            args.samples,
            args.steps,
            audio_input,
            args.noise_level
        ), seed
    
    audio_source = load_audio(inference.device, args.input[0], args.sr)
    audio_target = load_audio(inference.device, args.input[1], args.sr)
    
    audio_source_size = audio_source.size(dim=1)
    audio_target_size = audio_target.size(dim=1)
    
    min_length_multiplier = math.ceil(max([audio_source_size, audio_target_size]) / args.spc)
    if (args.length_multiplier == -1):
        audio_source = torch.nn.functional.pad(audio_source, (0, args.spc * max([min_length_multiplier, args.length_multiplier]) - audio_source_size), "constant", 0)
        audio_target = torch.nn.functional.pad(audio_target, (0, args.spc * max([min_length_multiplier, args.length_multiplier]) - audio_target_size), "constant", 0)
    else:
        audio_source = audio_source[:,:args.spc * args.length_multiplier]
        audio_target = audio_target[:,:args.spc * args.length_multiplier]
    
    return inference.generate_interpolation(
        args.samples,
        args.steps,
        audio_source,
        audio_target,
        args.noise_level
    ), seed


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


def parse_cli_args():
    parser = argparse.ArgumentParser()

    # args for model
    parser.add_argument(
        "--ckpt",
        metavar="CHECKPOINT",
        type=str,
        default="models/model.ckpt",
        help="path to the model checkpoint file to be used (default: models/model.ckpt)",
    )
    parser.add_argument(
        "--spc", 
        metavar="SAMPLES_PER_CHUNK",
        type=int, 
        default=65536, 
        help="the samples per chunk of the model (default: 65536)"
    )
    parser.add_argument(
        "--sr", 
        metavar="SAMPLE_RATE",
        type=int, 
        default=48000, 
        help="the samplerate of the model (default: 48000)"
    )

    # args for generation
    parser.add_argument(
        "--out_path",
        metavar="OUTPUT_PATH",
        type=str,
        default="audio_out",
        help="The path to the folder for the samples to be saved in (default: audio_out)",
    )
    parser.add_argument(
        "--length_multiplier",
        metavar="LENGTH",
        type=int,
        default=-1,
        help="The sample length multiplier for generate_variation (default: 1)",
    )
    parser.add_argument(
        "--input_sr",
        metavar="SAMPLE_RATE",
        type=int,
        default=44100,
        help="The sample rate of the input audio file specified in --input (default: 44100)",
    )
    parser.add_argument(
        "--noise_level", 
        metavar="NOISE_LEVEL",
        type=float, 
        default=0.7, 
        help="The noise level for generate_variation (default: 0.7)"
    )
    parser.add_argument(
        "--steps", 
        metavar="STEPS",
        type=int, 
        default=25, 
        help="The number of sampling steps (default: 25)"
    )
    parser.add_argument(
        "--samples",
        metavar="SAMPLES",
        type=int,
        default=1,
        help="The number of samples to generate per batch (default: 1)",
    )
    parser.add_argument(
        "--batches",
        metavar="BATCHES",
        type=int,
        default=1,
        help="The number of sample batches to generate (default: 1)",
    )
    parser.add_argument(
        "--seed", 
        metavar="SEED",
        type=int, 
        default=-1, 
        help="The random seed (default: -1)"
    )
    parser.add_argument(
        "--input",
        metavar="INPUT",
        type=str,
        default=None,
        nargs = '+',
        help="Path to the audio to be used for generate_variation or interpolations. If omitted, audio will be generated using random noise.",
    )

    # audio post-processing arguments
    parser.add_argument(
        "--remove_dc_offset",
        action="store_true",
        default=False,
        help="When this flag is set, a high pass filter will be applied to the input audio to remove DC offset.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="When this flag is set, output audio samples will be normalized.",
    )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        default=False,
        help="When this flag is set, processing will be done on the CPU.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        default=False,
        help="When this flag is set, the containing folder will be opened once your samples are saved.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
