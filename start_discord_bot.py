import os
import argparse
from dotenv import load_dotenv
from discord_bot.config import BotConfig
from discord_bot.bot import Bot


def main():
    token = load_env_vars()
    args = parse_cli_args()
    config = BotConfig(args.config_path)

    bot = Bot(token, args, config)
    bot.start()


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
        "--config_path",
        type=str,
        default="bot_config.ini",
        help="the path to the bot config file",
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
