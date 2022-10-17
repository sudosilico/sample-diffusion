import os
import torch
import discord
import multiprocessing as mp
import itertools
from multiprocessing.managers import SyncManager
from discord_bot.config import BotConfig
from discord_bot.models_metadata import ModelsMetadata
from discord_bot.request import DiffusionRequest
from discord_bot.response import handle_generation_response
from discord_bot.ui.regeneration import RegenerationUIView
from discord_bot.ui.variation import VariationUIView


def create_bot_with_commands(
    manager: SyncManager,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    args,
    config: BotConfig,
):
    bot = discord.Bot()

    models_metadata = ModelsMetadata(args.models_path)

    @bot.event
    async def on_ready():
        print(f"Bot '{bot.user}' has connected!")

    ckpt_paths = models_metadata.ckpt_paths.copy()

    def autocomplete(ctx):
        return ckpt_paths

    
    @bot.message_command(name="Regenerate...")
    async def regenerate(
        ctx: discord.commands.context.ApplicationContext,
        message: discord.message.Message,
    ):
        if message.author != bot.user or message.embeds is None or len(message.embeds) == 0:
            await ctx.respond("_Error: **Regenerate...** can only be performed on generations made by this bot._", ephemeral=True)
            return

        embed = message.embeds[0]

        model_ckpt = None
        seed = None
        steps = None
        length_multiplier = None
        samples = None

        for field in embed.fields:
            if field.name == "Model":
                model_ckpt = field.value
            elif field.name == "Seed":
                seed = int(field.value)
            elif field.name == "Steps":
                steps = int(field.value)
            elif field.name == "Length Multiplier":
                length_multiplier = int(field.value) if field.value != "Full Length" else -1
            elif field.name == "Samples":
                samples = int(field.value)

        # check if we're regenerating a variation or a generation
        if length_multiplier is not None:
            await ctx.respond("Error: **Regenerate...** can only be performed on generations, not variations.", ephemeral=True)
            return

        view = RegenerationUIView(
            models_metadata=models_metadata, 
            sync_manager=manager,
            request_queue=request_queue,
            response_queue=response_queue,
            seed=seed,
            steps=steps,
            samples=samples,
            model_ckpt=model_ckpt,
        )

        view.interaction = await ctx.respond(
            view=view,
            ephemeral=True,
            embed=view.get_embed(),
        )

            
    variation_id_iter = itertools.count()

    def next_variation_id():
        return next(variation_id_iter)

    @bot.message_command(name="Generate variation...")
    async def generate_variation(
        ctx: discord.commands.context.ApplicationContext,
        message: discord.message.Message,
    ):
        if message.attachments is None:
            await ctx.respond("_Error: You must choose a `.wav` file to generate a variation of._", ephemeral=True)
            return

        if len(message.attachments) == 0:
            await ctx.respond("_Error: You can only generate variations of `.wav` files._", ephemeral=True)
            return

        if len(message.attachments) > 1:
            # TODO: Add a sample selector view when used on a message with multiple attachments
            await ctx.respond("_Error: You can only generate variations of one `.wav` file._", ephemeral=True)
            return

        file = message.attachments[0]

        if file.content_type != "audio/x-wav":
            await ctx.respond("_Error: You can only generate variations of `.wav` files._", ephemeral=True)
            return
        
        max_file_size_bytes = config.get_int(ctx, "max_file_size_bytes", 600_000)
        if file.size > max_file_size_bytes:
            await ctx.respond(f"Error: File too large. WAV must be less than {max_file_size_bytes} bytes", ephemeral=True)
            return

        containing_folder = os.path.join(args.output_path, "variations")

        if not os.path.exists(containing_folder):
            os.makedirs(containing_folder)

        variation_id = next_variation_id()

        file_name, file_extension = os.path.splitext(file.filename)
        full_file_name = f"{variation_id}__{file_name}{file_extension}"
        file_path = os.path.join(containing_folder, full_file_name)

        print(f"Saving attachment to '{file_path}'")

        await file.save(file_path)

        view = VariationUIView(
            models_metadata, 
            file_path=file_path,
            file_name=file_name,
            sync_manager=manager,
            request_queue=request_queue,
            response_queue=response_queue
        )
        view.id = variation_id
        view.interaction = await ctx.respond(
            view=view,
            ephemeral=True,
            embed=view.get_embed(),
        )


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
            default=-1,
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
            max_value=int(config.get("admin", "max_steps")),
            default=25,
        ) = 25,
    ):
        # make sure model exists
        if not os.path.exists(os.path.join(args.models_path, model)):
            await ctx.respond(f"Error: Model '{model}' not found.")
            return

        seed = seed if seed != -1 else torch.seed()
        seed = seed % 4294967295

        max_samples = config.get_int(ctx, "max_samples")
        if samples < 1 or samples > max_samples:
            await ctx.respond(
                f"Error: Invalid number of samples: {samples}. Must be within the range [1, {max_samples}]."
            )
            return

        max_steps = config.get_int(ctx, "max_steps")
        if steps < 1 or steps > max_steps:
            await ctx.respond(
                f"Error: Invalid number of steps: {steps}. Must be within the range [1, {max_steps}]."
            )
            return

        start_event = manager.Event()
        done_event = manager.Event()
        progress_queue = manager.Queue()

        request = DiffusionRequest(
            model=model,
            seed=seed,
            samples=samples,
            steps=steps,
            start_event=start_event,
            done_event=done_event,
            progress_queue=progress_queue,
        )
        request_queue.put(request)

        original_message = await ctx.respond(
            f"Your request has been queued."
        )

        await handle_generation_response(
            ctx=ctx, 
            start_event=start_event,
            done_event=done_event, 
            progress_queue=progress_queue,
            request=request,
            response_queue=response_queue,
            original_message=original_message,
        )


    return bot