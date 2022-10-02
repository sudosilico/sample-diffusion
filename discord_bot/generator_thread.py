import json
import os
from queue import Queue
import threading
import time
import torchaudio
import torch
from generate import save_audio
from sample_diffusion.inference import generate_audio

from sample_diffusion.model import (
    ModelInfo,
    instantiate_model,
    load_state_from_checkpoint,
)


class GeneratorThread:
    def __init__(self, output_path="outputs_from_discord_bot"):
        self.output_path = output_path
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.running = False
        self.request_queue = Queue()

    def start(self):
        self.running = True
        self.thread.start()

    def add_request(self, request):
        self.request_queue.put(request)

    def run(self):
        while self.running:
            if not self.request_queue.empty():
                request_count = self.request_queue.qsize()
                print(f"There are currently {request_count} requests in the queue.")

                request = self.request_queue.get()

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
            else:
                time.sleep(1)

    def load_model(self, ckpt="models/model.ckpt", sample_rate=48000, chunk_size=65536):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_ph = instantiate_model(chunk_size, sample_rate)
        model = load_state_from_checkpoint(device, model_ph, ckpt)

        self.sample_rate = sample_rate
        self.model_info = ModelInfo(model, device, chunk_size)
