import asyncio
from multiprocessing.managers import SyncManager
import os
import torch
import discord
import multiprocessing as mp
import itertools
from discord_bot.config import BotConfig
from discord_bot.models_metadata import ModelsMetadata


class DiffusionRequest:
    id_iterator = itertools.count()

    def __init__(self, model: str, seed: int, samples: int, steps: int, done_event):
        self.id = next(DiffusionRequest.id_iterator)

        self.model = model
        self.seed = seed
        self.samples = samples
        self.steps = steps
        self.done_event = done_event


class DiffusionResponse:
    def __init__(self, request: DiffusionRequest, files: list[str], seed: int):
        self.request = request
        self.files = files
        self.seed = seed


def create_bot_with_commands(manager: SyncManager, request_queue: mp.Queue, response_queue: mp.Queue, args, config: BotConfig):
    bot = discord.Bot()

    models_metadata = ModelsMetadata(args.models_path)
    
    @bot.event
    async def on_ready():
        print(f"Bot '{bot.user}' has connected!")

    ckpt_paths = models_metadata.ckpt_paths.copy()
    def autocomplete(ctx):
        return ckpt_paths

    @bot.command(name="generate", help="Generate a sample from the model.")
    async def generate(
        ctx: discord.commands.context.ApplicationContext,
        model: discord.Option(
            str,
            "The Dance Diffusion model file to use for generation.",
            autocomplete=autocomplete,
        ),
        seed: discord.Option(
            int, 
            "The random seed. Use -1 or leave this out for a random seed.",
            default=-1
        ) = -1,
        samples: discord.Option(
            int,
            "The number of samples to generate.",
            min_value=1,
            max_value=10,
            default=1,
        ) = 1,
        steps: discord.Option(
            int, 
            "The number of steps to perform.", 
            min_value=1, 
            max_value=int(config.get('admin', 'max_steps')), 
            default=25
        ) = 25,
    ):

        # make sure model exists
        if not os.path.exists(os.path.join(args.models_path, model)):
            await ctx.respond(f"Error: Model '{model}' not found.")
            return

        seed = seed if seed != -1 else torch.seed()
        seed = seed % 4294967295

        max_samples = config.get_int(ctx, 'max_samples')
        if samples < 1 or samples > max_samples:
            await ctx.respond(
                f"Error: Invalid number of samples: {samples}. Must be within the range [1, {max_samples}]."
            )
            return

        max_steps = config.get_int(ctx, 'max_steps')
        if steps < 1 or steps > max_steps:
            await ctx.respond(
                f"Error: Invalid number of steps: {steps}. Must be within the range [1, {max_steps}]."
            )
            return

        done_event = manager.Event()

        request = DiffusionRequest(
            model=model,
            seed=seed,
            samples=samples,
            steps=steps,
            done_event=done_event,
        )
        request_queue_size = request_queue.qsize()
        request_queue.put(request)

        await ctx.respond(f"Your request has been queued. There are ~{request_queue_size} tasks ahead of yours in the queue.")

        current_id = request.id

        while not done_event.is_set():
            await asyncio.sleep(0.1)

        response: DiffusionResponse = response_queue.get(block=True)
        response_id = response.request.id
        
        if current_id == response_id:
            files = []
            for file in response.files:
                files.append(discord.File(file))

            if len(response.files) > 1:
                message = "Your samples are ready:"
            else:
                message = "Your sample is ready:"

            message += f"\n**Seed:** {seed}"
            message += f"\n**Samples:** {request.samples}"
            message += f"\n**Steps:** {request.steps}"
            message += f"\n**Model:** {model}\n"

            await ctx.send(
                files=files, content=f"{ctx.author.mention} {message}"
            )
        else:
            err = f"Internal error: ID mismatch. Got a response for ({response.request.id}) when processing ({request.id})."
            print(err)
            await ctx.send("Internal error. Please try again later.")


    return bot
