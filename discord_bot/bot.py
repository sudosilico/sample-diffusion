import multiprocessing as mp
from multiprocessing.managers import SyncManager
import os
import torchaudio
from discord_bot.commands import DiffusionRequest, DiffusionResponse, create_bot_with_commands
from discord_bot.config import BotConfig
from sample_diffusion.model import Model


def diffusion_process(
    request_queue: mp.Queue, 
    response_queue: mp.Queue, 
    args,
    config: BotConfig
):
    model = Model()
    
    while True:
        request: DiffusionRequest = request_queue.get(block=True)
        model_path = os.path.join(args.models_path, request.model)

        print(f"Loading model from {model_path}")
        model.load(model_path)
        print("Model loaded")

        sample_pural = "" if request.samples == 1 else "s"
        print(f"Generating {request.samples} sample{sample_pural} with seed {request.seed}")

        audio_out, seed = model.process_unconditional(seed=request.seed, samples=request.samples, steps=request.steps)
        print("Done generating. Saving audio...")

        output_path = args.output_path

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        model_str = f"{request.model}_{seed}_{request.samples}_{request.steps}"
        request_path = os.path.join(output_path, "generations", model_str)

        if not os.path.exists(request_path):
            os.makedirs(request_path)

        files = []

        for i, audio in enumerate(audio_out):
            output_path = os.path.join(request_path, f"sample_{i + 1}.wav")
            files.append(output_path)
            output = audio.cpu()
            torchaudio.save(output_path, output, model.sample_rate)

        print(f"Your files are located here: {request_path}")

        # open the request_path folder in a cross-platform way
        if args.open:
            if os.name == "nt":
                os.startfile(request_path)
            elif os.name == "posix":
                os.system(f"open {request_path}")

        response = DiffusionResponse(request, files, seed)
        response_queue.put(response)

        request.done_event.set()


class Bot:
    manager: SyncManager
    request_queue: mp.Queue
    response_queue: mp.Queue
    config: BotConfig

    def __init__(self, token: str, args, config: BotConfig):
        self.token = token
        self.args = args
        self.config = config

    def start(self):
        mp.set_start_method('spawn')

        self.manager = mp.Manager()
        self.request_queue = self.manager.Queue()
        self.response_queue = self.manager.Queue()

        self.diffusion_process = mp.Process(target=diffusion_process, args=(self.request_queue, self.response_queue, self.args, self.config))

        print("Starting diffusion process from main thread...")
        self.diffusion_process.start()

        print("Starting bot process from main thread...")
        bot = create_bot_with_commands(self.manager, self.request_queue, self.response_queue, self.args, self.config)
        bot.run(self.token)