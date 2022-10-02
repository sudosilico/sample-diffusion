import os, argparse
from dotenv import load_dotenv
from discord_bot.bot import DanceDiffusionDiscordBot

def main():
    token = load_env_vars()
    args = parse_cli_args()

    start_bot(token, args)


def start_bot(token, args):
    bot = DanceDiffusionDiscordBot()

    bot.load_model(ckpt=args.ckpt, sample_rate=args.sample_rate, chunk_size=args.chunk_size)

    bot.start(token)


def load_env_vars():
    load_dotenv()

    token = os.getenv("DISCORD_BOT_TOKEN")

    if token == None:
        print(
            "Error: No token found. Environment variable DISCORD_BOT_TOKEN must be set."
        )
        exit(1)

    return token


def parse_cli_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/model.ckpt",
        help="path to the model to be used",
    )

    parser.add_argument(
        "--chunk_size", type=int, default=65536, help="the chunk size (in samples) of the model"
    )
    parser.add_argument(
        "--sample_rate", type=int, default=48000, help="the samplerate of the model"
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
