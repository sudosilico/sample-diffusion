import multiprocessing as mp
from discord_bot.config import BotConfig
import os
import torchaudio
from discord_bot.config import BotConfig
from discord_bot.request import DiffusionRequest
from discord_bot.response import DiffusionResponse
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
        print(
            f"Generating {request.samples} sample{sample_pural} with seed {request.seed}"
        )

        # callback({'x': x, 'i': i, 't': steps[i], 'pred': pred})
        def callback(progress_vals):
            progress_float = (int(progress_vals["i"]) + 1) / request.steps
            progress = int(progress_float * 100)
            
            request.progress_queue.put(progress)

        request.start_event.set()

        if request.gen_type == "unconditional":
            audio_out, seed = model.process_unconditional(
                seed=request.seed, 
                samples=request.samples, 
                steps=request.steps,
                callback=callback,
            )
        elif request.gen_type == "variation":
            audio_out, seed = model.process_variation(
                audio_path=request.input,
                noise_level=request.noise_level,
                length_multiplier=request.length_multiplier,
                seed=request.seed, 
                samples=request.samples, 
                steps=request.steps,
                callback=callback,
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
