import json
from queue import Queue
import threading
import time
import discord
import os
from discord_bot.generator_thread import GeneratorThread


class DanceDiffusionDiscordBot:
    def __init__(self, config):
        self.config = config

        bot = discord.Bot()

        @bot.event
        async def on_ready():
            print(f"{bot.user} has connected!")

        @bot.slash_command()
        async def generate(ctx, noise_level: float = 0.7, n_steps: int = 25):
            # for key, value in ctx.__dict__.items():
            #     print(f"{key}: {value}")

            request = {
                "noise_level": noise_level,
                "n_steps": n_steps,
            }

            self.generator_thread.add_request(request)

            await ctx.respond(f"Request added to queue!")
            

        @bot.message_command(name="Generate variation")
        async def generate_variation(ctx, message):
            print(f"Executing in thread {threading.get_ident()}")

            await ctx.respond(
                f"{ctx.author.mention} says hello to {message.author.name}!"
            )

        self.bot = bot
        self.generator_thread = GeneratorThread()

    def start(self, token):
        self.generator_thread.start()
        self.bot.run(token)
