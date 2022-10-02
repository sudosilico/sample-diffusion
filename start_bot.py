import os
from dotenv import load_dotenv
from discord_bot.bot import DanceDiffusionDiscordBot


def main():
    token = load_env_vars()
    config = load_config()

    bot = DanceDiffusionDiscordBot(config)
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


def load_config():
    pass


if __name__ == "__main__":
    main()
