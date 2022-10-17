import itertools


class DiffusionRequest:
    id_iterator = itertools.count()

    def __init__(
        self, 
        model: str, 
        seed: int, 
        samples: int, 
        steps: int, 
        start_event,
        done_event,
        progress_queue,
        gen_type: str = "unconditional",
        input: str = None,
        input_name: str = None,
        noise_level: float = 0.7,
        length_multiplier: int = -1,
    ):
        self.id = next(DiffusionRequest.id_iterator)

        self.model = model
        self.seed = seed
        self.samples = samples
        self.steps = steps
        self.start_event = start_event
        self.done_event = done_event
        self.progress_queue = progress_queue
        self.gen_type = gen_type
        self.input = input
        self.input_name = input_name
        self.noise_level = noise_level
        self.length_multiplier = length_multiplier
