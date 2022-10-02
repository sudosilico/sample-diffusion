import json
from queue import Queue
import threading
import time
import discord
import asyncio
import os
from discord_bot.generator_thread import GeneratorThread
from discord_bot.thread_dispatcher import ThreadDispatcher


class DanceDiffusionDiscordBot:
    def __init__(self):
        self.dispatcher = ThreadDispatcher(daemon=False)
        self.dispatcher.start()

        self.dispatcher.invoke(self._initialize)

    def _initialize(self):
        bot = discord.Bot()

        @bot.event
        async def on_ready():
            print(f"{bot.user} has connected!")

        @bot.slash_command()
        async def generate(ctx, seed: int = -1, samples: int = 1, steps: int = 25):
            class GeneratorRequest(object):
                pass

            async def on_completed(sample_paths):
                for sample_path in sample_paths:
                    await ctx.respond(file=discord.File(sample_path))

            def oncompleted(sample_paths):
                loop = asyncio.get_event_loop()
                coroutine = on_completed(sample_paths)
                loop.run_until_complete(coroutine)
                
            request = GeneratorRequest()
            request.seed = seed
            request.samples = samples
            request.steps = steps
            request.oncompleted = lambda sample_paths : self.dispatcher.invoke(lambda : oncompleted(sample_paths))

            self.generator_thread.add_request(request)

            await ctx.respond(f"Request added to queue!")
            

        @bot.message_command(name="Generate variation")
        async def generate_variation(ctx, message):
            await ctx.respond(
                f"{ctx.author.mention} says hello to {message.author.name}!"
            )

        self.bot = bot
        self.generator_thread = GeneratorThread()
        
    def load_model(self, ckpt="models/model.ckpt", sample_rate=48000, chunk_size=65536):
        self.generator_thread.load_model(ckpt, sample_rate, chunk_size)

    def start(self, token):
        self.generator_thread.start()

        print("Starting discord bot...")
        self.bot.run(token)
