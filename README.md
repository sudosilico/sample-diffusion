# sample-diffusion

🚧 WIP 🚧

This contains the python project used by the [main web application](https://github.com/sudosilico/sample-diffusion-app), a sample generator that uses [Dance Diffusion](https://github.com/Harmonai-org/sample-generator).

Includes:

- `generate.py` script for using models to generate samples via CLI
- `server.py` script for starting a socket.io server, used by the web application to process the generation request queue and send progress updates

## Generator Guide

There is a [quick guide on using `generate.py`](https://github.com/sudosilico/sample-diffusion/blob/main/generator.md) to generate audio samples from your models.