import multiprocessing as mp
from multiprocessing.managers import SyncManager
import os
import torchaudio
from discord_bot.commands import create_bot_with_commands
from discord_bot.config import BotConfig
from discord_bot.diffusion import diffusion_process
#from dance_diffusion.model import Model


class Bot:
    def __init__(self, token: str, args, config: BotConfig):
        self.token = token
        self.args = args
        self.config = config

    def start(self):
        mp.set_start_method("spawn")

        self.manager = mp.Manager()
        self.request_queue = self.manager.Queue()
        self.response_queue = self.manager.Queue()

        self.diffusion_process = mp.Process(
            target=diffusion_process,
            args=(self.request_queue, self.response_queue, self.args, self.config),
        )

        print("Starting diffusion process from main thread...")
        self.diffusion_process.start()

        print("Starting bot process from main thread...")
        bot = create_bot_with_commands(
            self.manager,
            self.request_queue,
            self.response_queue,
            self.args,
            self.config,
        )
        bot.run(self.token)
