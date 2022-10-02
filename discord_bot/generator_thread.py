import json
import os
from queue import Queue
import threading
import time
import torchaudio
import torch
from discord_bot.thread_dispatcher import ThreadDispatcher
from generate import save_audio
from sample_diffusion.inference import generate_audio

from sample_diffusion.model import (
    ModelInfo,
    instantiate_model,
    load_state_from_checkpoint,
)


class GeneratorThread:
    def __init__(self, output_path="outputs_from_discord_bot"):
        #self.dispatcher = ThreadDispatcher(mode="asyncio")
        #self.dispatcher.start()
        self.request_queue = Queue()

        self.output_path = output_path

    def add_request(self, request):
        self.request_queue.put(request)

    def process_request(self, request):
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