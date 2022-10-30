import math
import multiprocessing as mp
from diffusion_api.inference import Inference
from diffusion_api.sampler import ImprovedPseudoLinearMultiStep
from diffusion_api.schedule import CrashSchedule
from diffusion_api.util import set_seed
from discord_bot.config import BotConfig
import os
import torchaudio
import torch
from discord_bot.config import BotConfig
from discord_bot.models_metadata import ModelsMetadata
from discord_bot.request import DiffusionRequest
from discord_bot.response import DiffusionResponse
from dance_diffusion.model import DanceDiffusionModel
from sample_diffusion.platform import get_torch_device_type
from sample_diffusion.util import load_audio


def diffusion_process(
    request_queue: mp.Queue, 
    response_queue: mp.Queue, 
    args, 
    config: BotConfig,
    models_metadata: ModelsMetadata
):
    model = None
    device_type = get_torch_device_type()
    device = torch.device(device_type)

    while True:
        request: DiffusionRequest = request_queue.get(block=True)
        model_path = os.path.join(args.models_path, request.model)

        print(f"Loading model from {model_path}")

        if model is None or model.model_path != model_path:
            
            meta = models_metadata.get_meta(request.model)

            if meta == None:
                print(f"Model {request.model} not found")
            else:
                model = DanceDiffusionModel(
                    device=device,
                    chunk_size=int(meta["chunk_size"]),
                    sample_rate=int(meta["sample_rate"])
                )

            model.load(model_path, model.chunk_size, model.sample_rate)
            
            print("Model loaded")
        else:
            print("Skipping model loading,", model.model_path, "is already loaded")


        sample_pural = "" if request.samples == 1 else "s"
        print(
            f"Generating {request.samples} sample{sample_pural} with seed {request.seed}"
        )

        def callback(progress_vals):
            progress_float = (int(progress_vals["i"]) + 1) / (request.steps - 2)
            progress = int(progress_float * 100)
            
            request.progress_queue.put(progress)

        request.start_event.set()

        scheduler = CrashSchedule(device)
        sampler = ImprovedPseudoLinearMultiStep(model)
        inference = Inference(device, model, sampler, scheduler)

        seed = set_seed(request.seed)

        if request.gen_type == "unconditional":
            audio_out = inference.generate_unconditional(
                request.samples,
                request.steps,
                callback=callback
            )
        elif request.gen_type == "variation":
            audio_input = load_audio(device, request.input, model.sample_rate)
            audio_input_size = audio_input.size(dim=1)
            min_length_multiplier = math.ceil(audio_input_size / model.chunk_size)

            length_multiplier = request.length_multiplier

            if (length_multiplier == -1):
                audio_input = torch.nn.functional.pad(audio_input, (0, model.chunk_size * max([min_length_multiplier, length_multiplier]) - audio_input_size), "constant", 0)
            else:
                audio_input = audio_input[:,:model.chunk_size * length_multiplier]

            audio_out = inference.generate_variation(
                audio_input=audio_input,
                batch_size=request.samples,
                noise_level=request.noise_level,
                steps=request.steps,
                callback=callback
            )

            if os.path.exists(request.input):
                os.remove(request.input)

        print("Done generating. Saving audio...")

        output_path = args.output_path

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        model_str = f"{request.model}_{seed}_{request.samples}_{request.steps}"
        type_folder = "generations" if request.gen_type == "unconditional" else "variations"
        request_path = os.path.join(output_path, type_folder, model_str)

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
