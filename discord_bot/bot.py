import json
from queue import Queue
import threading
import time
import torch
import torchaudio
import discord
import asyncio
import os
from sample_diffusion.inference import generate_audio
from sample_diffusion.model import ModelInfo, instantiate_model, load_state_from_checkpoint

def start_discord_bot(token, args, output_path="outputs_from_discord_bot", models_path="models", max_queue_size=10):
    bot = DanceDiffusionDiscordBot(output_path, models_path, max_queue_size)

    print(f"Loading model checkpoint '{args.ckpt}'...")
    bot.load_model(ckpt=args.ckpt, sample_rate=args.sample_rate, chunk_size=args.chunk_size)
    
    print("Starting discord bot...")
    bot.start(token)

class DanceDiffusionDiscordBot:
    def __init__(self, output_path, models_path, max_queue_size):
        self.output_path = output_path
        self.models_path = models_path
        self.ckpt_paths = ['models/glitch.ckpt', 'models/glitch_trim.ckpt']
        self.max_queue_size = max_queue_size
        self.ckpt = None

        self.discord_loop = asyncio.new_event_loop()
        self.generator_loop = asyncio.new_event_loop()
        self.generator_thread = threading.Thread(target=self.run_generator_thread)
        self.processing_tasks = []

        bot = discord.Bot()
        bot.loop = self.discord_loop

        @bot.event
        async def on_ready():
            print(f"{bot.user} has connected!")

        async def get_ckpt(ctx: discord.AutocompleteContext):
            """Return's A List Of Autocomplete Results"""
            return self.ckpt_paths # from your database

        @bot.slash_command()
        async def generate(
            ctx, 
            seed: int = -1, 
            samples: int = 1, 
            steps: int = 25, 
            model: discord.Option(str, "", autocomplete=get_ckpt) = "models/glitch_trim.ckpt"):

            class GeneratorRequest(object):
                pass

            if len(self.processing_tasks) >= self.max_queue_size:
                await ctx.respond(content=f"{ctx.author.mention} The queue is currently full. Please try again later.")
                return

            async def on_completed(sample_paths):
                files = []

                for sample_path in sample_paths:
                    files.append(discord.File(sample_path))

                if len(sample_paths) > 1:
                    message = "Your samples are ready"
                else:
                    message = "Your sample is ready"

                await ctx.send_followup(files=files, content=f"{ctx.author.mention} {message}:")

            discord_loop = self.discord_loop

            def oncompleted(sample_paths):
                coroutine = on_completed(sample_paths)
                discord_loop.create_task(coroutine)
                
            request = GeneratorRequest()
            request.seed = seed
            request.samples = samples
            request.steps = steps
            request.oncompleted = oncompleted

            coroutine = self.process_request(request)
            request.future = asyncio.run_coroutine_threadsafe(coroutine, self.generator_loop)
            self.processing_tasks.append(request.future)

            queue_size = len(self.processing_tasks)
            await ctx.respond(f"{ctx.author.mention} Your request has been added to the queue. There are currently {queue_size} tasks in the queue.")
            
        async def select_model(self, ctx, callback):
            class ModelSelectView(discord.ui.View):
                async def on_timeout(self):
                    for child in self.children:
                        child.disabled = True
                    await self.message.edit(content="You took too long!", view=self)

                @discord.ui.button(label="Button 1", row=0, style=discord.ButtonStyle.primary)
                async def first_button_callback(self, button, interaction):
                    await interaction.response.send_message("You pressed me!")

                @discord.ui.button(label="Button 2", row=0, style=discord.ButtonStyle.primary)
                async def second_button_callback(self, button, interaction):
                    await interaction.response.send_message("You pressed me!")

                @discord.ui.select(
                    placeholder = "Choose a Flavor!", # the placeholder text that will be displayed if nothing is selected
                    min_values = 1, # the minimum number of values that must be selected by the users
                    max_values = 1, # the maximum number of values that can be selected by the users
                    options = [ # the list of options from which users can choose, a required field
                        discord.SelectOption(
                            label="Vanilla",
                            description="Pick this if you like vanilla!"
                        ),
                        discord.SelectOption(
                            label="Chocolate",
                            description="Pick this if you like chocolate!"
                        ),
                        discord.SelectOption(
                            label="Strawberry",
                            description="Pick this if you like strawberry!"
                        )
                    ]
                )
                async def select_callback(self, select, interaction):
                    await callback()
                    await interaction.response.send_message(f"Awesome! I like {select.values[0]} too!")

            await ctx.send("Choose a flavor!", view=ModelSelectView())

        @bot.message_command(name="Generate variation")
        async def generate_variation(ctx, message):
            await ctx.respond(
                f"{ctx.author.mention} says hello to {message.author.name}!"
            )

        self.bot = bot

    def run_generator_thread(self):
        self.generator_loop.run_forever()
        
    def load_model(self, ckpt="models/model.ckpt", sample_rate=48000, chunk_size=65536):
        if self.ckpt == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_ph = instantiate_model(chunk_size, sample_rate)
            model = load_state_from_checkpoint(device, model_ph, ckpt)

            self.sample_rate = sample_rate
            self.model_info = ModelInfo(model, device, chunk_size)
            self.ckpt = ckpt
        else:
            self.sample_rate = sample_rate
            self.model_info.switch_models(ckpt, sample_rate, chunk_size)
            self.ckpt = ckpt

    def start(self, token):
        self.generator_thread.start()
        self.bot.run(token)

    async def wait_for_completed(self):
        while True:
            await asyncio.sleep(1)

    async def process_request(self, request):
        print("Processing request...")

        seed = request.seed
        samples = request.samples
        steps = request.steps
        oncompleted = request.oncompleted

        print("Generating audio...")
        print(f"Seed: {seed}, Samples: {samples}, Steps: {steps}")

        audio_out, seed = generate_audio(seed, samples, steps, self.model_info)

        print("Done. Exporting audio...")

        samples_output_path = os.path.join(self.output_path, f"{seed}_{steps}")

        if not os.path.exists(samples_output_path):
            os.makedirs(samples_output_path)

        sample_paths = []

        # save audio samples to files
        for ix, sample in enumerate(audio_out):
            output_file = os.path.join(
                samples_output_path, f"sample_{ix + 1}.wav"
            )
            open(output_file, "a").close()
            output = sample.cpu()
            torchaudio.save(output_file, output, self.sample_rate)
            sample_paths.append(output_file)

        print("Saved audio samples to:")
        print(samples_output_path)
        
        oncompleted(sample_paths)
        self.processing_tasks.remove(request.future)

