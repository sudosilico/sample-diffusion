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
        async def generate(ctx, seed: int = -1, samples: int = 1, steps: int = 25):
            class GeneratorRequest(object):
                pass

            def oncompleted(sample_paths):
                for sample_path in sample_paths:
                    ctx.respond(file=discord.File(sample_path))
                
            request = GeneratorRequest()
            request.seed = seed
            request.samples = samples
            request.steps = steps
            request.oncompleted = oncompleted

            self.generator_thread.add_request(request)

            await ctx.respond(f"Request added to queue!")
            

        @bot.message_command(name="Generate variation")
        async def generate_variation(ctx, message):
            await ctx.respond(
                f"{ctx.author.mention} says hello to {message.author.name}!"
            )

        self.bot = bot
        self.generator_thread = GeneratorThread()

    def start(self, token):
        self.generator_thread.start()
        self.bot.run(token)
