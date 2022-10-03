import json
import threading
import torch
import torchaudio
import discord
import functools
import asyncio
import os
from sample_diffusion.inference import generate_audio
from sample_diffusion.model import ModelInfo, instantiate_model, load_state_from_checkpoint

def start_discord_bot(token, args):
    models_path = args.models_path
    output_path = args.output_path
    max_queue_size = args.max_queue_size

    bot = DanceDiffusionDiscordBot(output_path, models_path, max_queue_size)

    print("Starting discord bot...")
    bot.start(token)

def load_models_path(models_path):
    ckpt_paths = []

    if not os.path.exists(models_path):
            os.makedirs(models_path)

    for root, dirs, files in os.walk(models_path):
        for file in files:
            if file.endswith(".ckpt"):
                ckpt_paths.append(file)

    print("Found the following model checkpoints:")
    print(ckpt_paths)

    models_json_path = os.path.join(models_path, "models.json")
    models_metadata = []

    if os.path.exists(models_json_path):
        with open(models_json_path, "r") as f:
            models_meta = json.load(f)
            models_metadata = models_meta["models"]

            def json_contains_checkpoint(path):
                for model in models_metadata:
                    if model["path"] == path:
                        return True
                
                return False

            # make sure every model on disk has a corresponding entry in the json file
            for checkpoint in ckpt_paths:
                if not json_contains_checkpoint(checkpoint):
                    models_metadata.append({
                        "path": checkpoint,
                        "name": checkpoint,
                        "sample_rate": 48000,
                        "chunk_size": 65536,
                        "description": "",
                    })
                    print(f"WARNING: Creating models.json with default sample_rate and chunk_size values for '{checkpoint}'.")
                    print("You may need to edit this file to change these to the correct values.")
    else:
        for checkpoint in ckpt_paths:
            model = {
                "path": checkpoint,
                "name": checkpoint,
                "sample_rate": 48000,
                "chunk_size": 65536,
                "description": "",
            }

            models_metadata.append(model)
            print(f"WARNING: Creating models.json with default sample_rate and chunk_size values for '{checkpoint}'.")
            print("You may need to edit this file to change these to the correct values.")

    with open(models_json_path, "w") as f:
        json.dump({"models": models_metadata}, f, indent=4)

    return ckpt_paths, models_metadata

class DanceDiffusionDiscordBot:
    def __init__(self, output_path, models_path, max_queue_size):
        self.output_path = output_path
        self.models_path = models_path

        ckpt_paths, models_metadata = load_models_path(models_path)
        self.ckpt_paths = ckpt_paths
        self.models_metadata = models_metadata

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
            model: discord.Option(str, "", autocomplete=get_ckpt),
            seed: int = -1, 
            samples: int = 1, 
            steps: int = 25):

            model_exists = os.path.exists(os.path.join(self.models_path, model))

            if not model_exists:
                await ctx.respond(content=f"{ctx.author.mention} Invalid model. Please choose one of the following models: {self.ckpt_paths}")
                return

            use_seed = seed if seed != -1 else torch.seed()

            class GeneratorRequest(object):
                pass

            if len(self.processing_tasks) >= self.max_queue_size:
                await ctx.respond(content=f"{ctx.author.mention} The queue is currently full. Please try again later.")
                return

            async def on_completed(sample_paths, request, seed):
                files = []

                for sample_path in sample_paths:
                    files.append(discord.File(sample_path))

                if len(sample_paths) > 1:
                    message = "Your samples are ready:"
                else:
                    message = "Your sample is ready:"

                message += f"\n**Seed:** {seed}"
                message += f"\n**Samples:** {request.samples}"
                message += f"\n**Steps:** {request.steps}"
                message += f"\n**Model:** {model}\n"

                await ctx.respond(files=files, content=f"{ctx.author.mention} {message}")

            discord_loop = self.discord_loop

            request = GeneratorRequest()

            def oncompleted(sample_paths, req, seed):
                coroutine = on_completed(sample_paths, req, seed)
                discord_loop.create_task(coroutine)

            request.seed = use_seed
            request.samples = samples
            request.steps = steps
            request.ckpt = os.path.join(self.models_path, model)
            request.oncompleted = oncompleted

            print("Getting model metadata...")
            sample_rate, chunk_size = self.get_model_meta(model)

            request.sample_rate = sample_rate
            request.chunk_size = chunk_size            

            queue_size = len(self.processing_tasks)
            await ctx.respond(f"{ctx.author.mention} Your request has been added to the queue. There are currently {queue_size} tasks ahead of you in the queue.")

            coroutine = self.process_request(request)
            request.future = asyncio.run_coroutine_threadsafe(coroutine, self.generator_loop)
            self.processing_tasks.append(request.future)


            
        # @bot.message_command(name="Generate variation")
        # async def generate_variation(ctx, message):
        #     await ctx.respond(
        #         f"{ctx.author.mention} says hello to {message.author.name}!"
        #     )

        self.bot = bot

    def run_generator_thread(self):
        self.generator_loop.run_forever()
        
    def load_model(self, ckpt, sample_rate, chunk_size):
        print("Loading model...")
        if self.ckpt == None:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"torch device: {device_type}")
            device = torch.device(device_type)
            model_ph = instantiate_model(chunk_size, sample_rate)
            model = load_state_from_checkpoint(device, model_ph, ckpt)

            self.sample_rate = sample_rate
            self.model_info = ModelInfo(model, device, chunk_size)
            self.ckpt = ckpt
        else:
            self.sample_rate = sample_rate

            if self.ckpt == ckpt:
                print(f"Model already loaded: '{ckpt}'.")
            else:
                self.model = self.model_info.switch_models(ckpt, sample_rate, chunk_size)
                self.ckpt = ckpt
                print(f"Model loaded: '{ckpt}'")

        

    def get_model_meta(self, ckpt):
        print("get_model_metadata")
        print(f"len(self.models_metadata): {len(self.models_metadata)}")

        for model_meta in self.models_metadata:
            if model_meta["path"] == ckpt:
                sample_rate = model_meta["sample_rate"]
                chunk_size = model_meta["chunk_size"]

                return sample_rate, chunk_size

        print(f"WARNING: Could not locate '{ckpt}' path in model metadata. Using default sample_rate and chunk_size.")
        return 48000, 65536
        
    def start(self, token):
        self.generator_thread.start()
        self.bot.run(token)

    async def wait_for_completed(self):
        while True:
            await asyncio.sleep(1)

    async def process_request(self, request):
        print("Processing request...")

        samples = request.samples
        steps = request.steps
        oncompleted = request.oncompleted
        sample_rate = request.sample_rate
        chunk_size = request.chunk_size
        ckpt = request.ckpt

        async def load_model(ckpt, sr, cs):
            model = self.load_model(request.ckpt, sample_rate, chunk_size)
            return model

        print("loading model...")
        #self.load_model(request.ckpt, sample_rate, chunk_size)

        if self.ckpt == None:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"torch device: {device_type}")
            device = torch.device(device_type)
            model_ph = instantiate_model(chunk_size, sample_rate)
            model = load_state_from_checkpoint(device, model_ph, ckpt)

            self.sample_rate = sample_rate
            self.model_info = ModelInfo(model, device, chunk_size)
            self.ckpt = ckpt
        else:
            self.sample_rate = sample_rate

            if self.ckpt == ckpt:
                print(f"Model already loaded: '{ckpt}'.")
            else:
                self.model = self.model_info.switch_models(ckpt, sample_rate, chunk_size)
                self.ckpt = ckpt
                print(f"Model loaded: '{ckpt}'")

        print("model loaded...")
        # request.future = asyncio.run_coroutine_threadsafe(coroutine, self.discord_loop)

        print("Generating audio...")
        print(f"Seed: {request.seed}, Samples: {samples}, Steps: {steps}")

        audio_out, seed = generate_audio(request.seed, samples, steps, self.model_info)

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

        if request.future in self.processing_tasks:
            self.processing_tasks.remove(request.future)

        oncompleted(sample_paths, request, seed)

