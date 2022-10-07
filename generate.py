import os, argparse
import torchaudio
import torch
import json
import time
from sample_diffusion.model import load_model
from sample_diffusion.inference import generate_audio, process_audio


def main():
    args = parse_cli_args()

    model_info = load_model(args)
    seed = args.seed if args.seed != -1 else torch.seed()

    start_time = time.process_time()

    for batch in range(args.n_batches):
        audio_out, seed = perform_batch(args, model_info, seed + batch)

        if args.remove_dc_offset:
            print("Filtering DC offset...")
            audio_out = remove_dc_offset(audio_out, args.sr)

        if args.normalize:
            print("Normalizing...")
            audio_out = normalize_audio(audio_out)

        save_audio(audio_out, args, seed, batch)

    end_time = time.process_time()
    elapsed = end_time - start_time

    print(
        f"Done! Generated {args.n_samples * args.n_batches} samples in {elapsed} seconds."
    )


def remove_dc_offset(audio_out, sample_rate):
    return torchaudio.functional.highpass_biquad(audio_out, sample_rate, 15, 0.707)


def normalize_audio(audio_out):
    return audio_out / torch.max(torch.abs(audio_out))


def perform_batch(args, model_info, seed):
    if args.input:
        return process_audio(
            args.input,
            args.input_sr,
            args.sample_length_multiplier,
            args.noise_level,
            seed,
            args.n_samples,
            args.n_steps,
            model_info,
        )

    return generate_audio(seed, args.n_samples, args.n_steps, model_info)


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

    if args.n_batches > 1:
        print(f"Finished batch {batch + 1} of {args.n_batches}.")

    print(f"\nYour samples are waiting for you here: {output_path}")

    if args.input:
        print(f"  Seed: {seed}, Steps: {args.n_steps}, Noise: {args.noise_level}\n")
    else:
        print(f"  Seed: {seed}, Steps: {args.n_steps}\n")


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
    if args.input:
        parent_folder = os.path.join(
            args.out_path, f"variations", f"{seed}_{args.n_steps}_{args.noise_level}"
        )
    else:
        parent_folder = os.path.join(
            args.out_path, f"generations", f"{seed}_{args.n_steps}"
        )

    return parent_folder


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
        "--sample_length_multiplier",
        metavar="LENGTH",
        type=int,
        default=1,
        help="The sample length multiplier for audio2audio (default: 1)",
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
        help="The noise level for audio2audio (default: 0.7)"
    )
    parser.add_argument(
        "--n_steps", 
        metavar="STEPS",
        type=int, 
        default=25, 
        help="The number of sampling steps (default: 25)"
    )
    parser.add_argument(
        "--n_samples",
        metavar="SAMPLES",
        type=int,
        default=1,
        help="The number of samples to generate per batch (default: 1)",
    )
    parser.add_argument(
        "--n_batches",
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
        default="",
        help="Path to the audio to be used for audio2audio. If omitted, audio will be generated using random noise.",
    )
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

    return parser.parse_args()


if __name__ == "__main__":
    main()
