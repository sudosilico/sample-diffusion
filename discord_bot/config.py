import configparser
import os
import discord

class BotConfig:
    def __init__(self, config_file: str):
        self._config = configparser.ConfigParser()

        if not os.path.exists(config_file):
            print(f"Config file not found: '{config_file}'")

        self._config.read(config_file)

    def get(
        self,
        category: str,
        key: str
    ):
        return self._config[category][key]

    def get_int(
        self, 
        ctx: discord.commands.context.ApplicationContext,
        key: str,
        default: int = None
    ):
        user_category = f"user:{ctx.author.id}"

        if user_category in self._config:
            if key in self._config[user_category]:
                val = self._config[user_category][key]
                return default if (val is None) else int(val)

        val = self._config.get(
            "admin" if self._is_admin(ctx) else "DEFAULT", 
            key
        )

        return default if (val is None) else int(val)

    def _is_admin(self, ctx: discord.commands.context.ApplicationContext):
        is_guild = ctx.guild is not None
        is_guild_admin = ctx.author.guild_permissions.administrator if is_guild else False

        return is_guild_admin