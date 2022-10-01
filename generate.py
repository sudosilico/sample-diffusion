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
        perform_batch(args, model_info, seed + batch, batch)

    end_time = time.process_time()
    elapsed = end_time - start_time

    print(
        f"Done! Generated {args.n_samples * args.n_batches} samples in {elapsed} seconds."
    )


def perform_batch(args, model_info, seed, batch):
    if args.input:
        audio_out, seed = process_audio(
            args.input,
            args.input_sr,
            args.sample_length_multiplier,
            args.noise_level,
            seed,
            args.n_samples,
            args.n_steps,
            model_info,
        )
        save_audio(audio_out, args, seed, batch)
    else:
        audio_out, seed = generate_audio(seed, args.n_samples, args.n_steps, model_info)
        save_audio(audio_out, args, seed, batch)


def save_audio(audio_out, args, seed, batch):
    output_path = get_output_folder(args, seed, batch)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    write_metadata(args, seed, batch, os.path.join(output_path, "meta.json"))

    for ix, sample in enumerate(audio_out):
        output_file = os.path.join(output_path, f"sample #{ix + 1}.wav")
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
        type=str,
        default="models/model.ckpt",
        help="path to the model to be used",
    )
    parser.add_argument(
        "--spc", type=int, default=65536, help="the samples per chunk of the model"
    )
    parser.add_argument(
        "--sr", type=int, default=48000, help="the samplerate of the model"
    )

    # args for generation
    parser.add_argument(
        "--out_path",
        type=str,
        default="audio_out",
        help="path to the folder for the samples to be saved in",
    )
    parser.add_argument(
        "--sample_length_multiplier",
        type=int,
        default=1,
        help="sample length multiplier for audio2audio",
    )
    parser.add_argument(
        "--input_sr",
        type=int,
        default=44100,
        help="samplerate of the input audio specified in --input",
    )
    parser.add_argument(
        "--noise_level", type=float, default=0.7, help="noise level for audio2audio"
    )
    parser.add_argument(
        "--n_steps", type=int, default=25, help="number of sampling steps"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to generate per batch",
    )
    parser.add_argument(
        "--n_batches",
        type=int,
        default=1,
        help="how many batches of samples to generate",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="the seed (for reproducible sampling)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="path to the audio to be used for audio2audio",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
