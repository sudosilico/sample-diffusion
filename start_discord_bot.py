import os, argparse
from dotenv import load_dotenv
from discord_bot.bot import start_discord_bot


def main():
    token = load_env_vars()
    args = parse_cli_args()

    start_discord_bot(token, args)


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
        "--models_path",
        type=str,
        default="models",
        help="path to the models folder",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs_from_discord_bot",
        help="path to the outputs folder",
    )
    parser.add_argument(
        "--max_queue_size",
        type=int,
        default=10,
        help="the maximum size of the request queue",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
