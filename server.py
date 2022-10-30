import os, argparse
import torchaudio
from dance_diffusion.model import load_model
from sample_diffusion.inference import generate_audio
from sample_diffusion.server import SocketIOServer


def main():
    args = parse_cli_args()

    model, device = load_model(args)

    server = create_server(args, model, device)
    server.start()


def create_server(args, model, device):

    server = SocketIOServer(
        args, host=args.ws_host, port=args.ws_port, website_url=args.public_url
    )

    def generate(generation_args, socketio):
        seed = generation_args.seed
        audio_out = generate_audio(
            generation_args, seed, args, device, model, server.progress_cb()
        )
        save_audio(generation_args, args, audio_out)

        pass

    server.ongenerate = generate

    return


def save_audio(generation_args, model_args, audio_out, base_out_path):
    output_path = get_output_path(generation_args, base_out_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for ix, sample in enumerate(audio_out):
        output_file = os.path.join(output_path, f"sample #{ix + 1}.wav")
        open(output_file, "a").close()
        output = sample.cpu()
        torchaudio.save(output_file, output, model_args.sr)

    print(f"Your samples are waiting for you here: {output_path}")


def get_output_path(generation_args, base_out_path):
    if generation_args.input:
        return os.path.join(
            base_out_path,
            "variations",
            f"{generation_args.seed}_{generation_args.steps}_{generation_args.noise_level}/",
        )

    return os.path.join(
        base_out_path,
        "generations",
        f"{generation_args.seed}_{generation_args.steps}/",
    )


def parse_cli_args():
    parser = argparse.ArgumentParser()

    # args for socketio server
    parser.add_argument(
        "--ws_host", type=str, default="localhost", help="host for the websocket server"
    )
    parser.add_argument(
        "--ws_port", type=int, default=5001, help="port for the websocket server"
    )
    parser.add_argument(
        "--public_url",
        type=str,
        default="http://localhost:3000",
        help="public url for the dashboard",
    )

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
    parser.add_argument(
        "--out_path",
        type=str,
        default="audio_out",
        help="path to the folder for the samples to be saved in",
    )

    # args for generation
    parser.add_argument(
        "--length_multiplier",
        type=int,
        default=1,
        help="sample length multiplier for generate_variation",
    )
    parser.add_argument(
        "--input_sr",
        type=int,
        default=44100,
        help="samplerate of the input audio specified in --input",
    )
    parser.add_argument(
        "--noise_level", type=float, default=0.7, help="noise level for generate_variation"
    )
    parser.add_argument(
        "--steps", type=int, default=25, help="number of sampling steps"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="how many samples to produce / batch size",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="the seed (for reproducible sampling)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="path to the audio to be used for generate_variation",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
