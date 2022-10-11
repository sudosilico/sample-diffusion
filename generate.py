import os, argparse
import torchaudio
import json
import time
from sample_diffusion.model import Model
from sample_diffusion.post_process import post_process_audio


def main():
    args = parse_cli_args()

    model = Model(force_cpu=args.force_cpu)
    model.load(
        model_path=args.ckpt,
        chunk_size=args.spc,
        sample_rate=args.sr,
    )

    start_time = time.process_time()

    for batch in range(args.batches):
        audio_out, seed = perform_batch(model, args.seed + batch, args)

        audio_out = post_process_audio(
            audio_out, 
            sample_rate=args.sr, 
            remove_dc_offset=args.remove_dc_offset, 
            normalize=args.normalize
        )

        save_audio(audio_out, args, seed, batch)

    end_time = time.process_time()
    elapsed = end_time - start_time

    print(
        f"Done! Generated {args.samples * args.batches} samples in {elapsed} seconds."
    )


def perform_batch(model: Model, seed, args):
    if args.input:
        return model.process_audio_file(
            audio_path=args.input,
            noise_level=args.noise_level,
            length_multiplier=args.length_multiplier,
            seed=seed,
            samples=args.samples,
            steps=args.steps,
        )

    return model.generate(
        seed=seed,
        samples=args.samples,
        steps=args.steps,
    )


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


    if args.input:
        print(f"  Seed: {seed}, Steps: {args.steps}, Noise: {args.noise_level}\n")
    else:
        print(f"  Seed: {seed}, Steps: {args.steps}\n")


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
            args.out_path, f"variations", f"{seed}_{args.steps}_{args.noise_level}"
        )
    else:
        parent_folder = os.path.join(
            args.out_path, f"generations", f"{seed}_{args.steps}"
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
        "--length_multiplier",
        metavar="LENGTH",
        type=int,
        default=-1,
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
        default="",
        help="Path to the audio to be used for audio2audio. If omitted, audio will be generated using random noise.",
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
        help="when this flag is used, the bot will open the output folder after generation",
    )


    return parser.parse_args()


if __name__ == "__main__":
    main()
