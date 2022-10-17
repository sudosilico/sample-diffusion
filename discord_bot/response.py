
import asyncio
import discord
from discord_bot.request import DiffusionRequest


class DiffusionResponse:
    def __init__(self, request: DiffusionRequest, files: "list[str]", seed: int):
        self.request = request
        self.files = files
        self.seed = seed


def make_progress_bar_string(progress, length=16):
    filled_length = int(length * progress / 100)
    unfilled_length = length - filled_length

    bar = "▓" * filled_length
    bar += "░" * unfilled_length
    return f"{bar}"


async def handle_generation_response(
    start_event,
    done_event, 
    progress_queue,
    request, 
    response_queue, 
    original_message=None, 
    ctx: discord.commands.context.ApplicationContext = None, 
    interaction: discord.Interaction = None,
    parent_view = None,
):
    current_id = request.id

    # wait for start event
    while not start_event.is_set():
        await asyncio.sleep(0.1)
    
    if ctx is None:
        msg = await interaction.followup.send(f"{interaction.user.mention} Starting generation...")
    else:
        msg = await ctx.send(f"{ctx.author.mention} Starting generation...")

    # update progress from progress_queue updates while generating
    done = False
    progress = 0
    
    update_embed = parent_view.get_embed() if parent_view is not None else None

    while not done:
        await asyncio.sleep(3)

        while not progress_queue.empty():
            progress = progress_queue.get()

        if progress == 100 or done_event.is_set():
            done = True
            break

        progress_bar_string = make_progress_bar_string(progress)

        if ctx is None:
            content = f"{interaction.user.mention} Generating...\n\n{progress_bar_string} {progress}%"
        else:
            content=f"{ctx.author.mention} Generating...\n\n{progress_bar_string} {progress}%"

        await msg.edit(
            embed=update_embed,
            content=content,
        )

    await msg.edit(content=f"Generation complete!")

    # wait for done event
    while not done_event.is_set():
        await asyncio.sleep(1)

    response: DiffusionResponse = response_queue.get(block=True)
    response_id = response.request.id

    if current_id != response_id:
        err = f"Internal error: ID mismatch. Got a response for ({response.request.id}) when processing ({request.id})."
        print(err)
        if ctx is None:
            await interaction.followup.send("Internal error. Please try again later.")
        else:
            await ctx.send("Internal error. Please try again later.")


    files = []
    for file in response.files:
        files.append(discord.File(file))

    if len(response.files) > 1:
        message = "Your samples are ready:"
    else:
        message = "Your sample is ready:"


    if ctx is None:
        await interaction.followup.send(
            embed=create_response_embed(response),
            files=files,
            content=f"{interaction.user.mention} {message}"
        )
    else:
        original_response = await original_message.original_response()
        await ctx.send(
            embed=create_response_embed(response),
            files=files,
            content=f"{ctx.author.mention} {message}",
            reference=original_response,
        )


def create_response_embed(response):
    gen_type = response.request.gen_type

    if gen_type == "variation":
        embed_title = f"Variation of `{response.request.input_name}` with options:"
    else:
        embed_title = "Generation options:"

    embed = discord.Embed(title=embed_title, color=0x01239B)

    if gen_type == "variation": 
        embed.add_field(name="Noise Level", value=response.request.noise_level)

    embed.add_field(name="Steps", value=response.request.steps)
    embed.add_field(name="Samples", value=response.request.samples)
    
    if gen_type == "variation": 
        embed.add_field(
            name="Length Multiplier", 
            value=response.request.length_multiplier if response.request.length_multiplier != -1 else "Full Length"
        )
        
    embed.add_field(name="Seed", value=response.request.seed)
    embed.add_field(name="Model", value=response.request.model)
    
    return embed
