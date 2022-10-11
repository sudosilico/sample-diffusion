# sample-diffusion

🚧 This project is early in development. Expect breaking changes! 🚧

This repository contains the python project that runs machine learning tasks for the [Sample Diffusion web application](https://github.com/sudosilico/sample-diffusion-app). It is used to generate audio samples using Harmonai [Dance Diffusion](https://github.com/Harmonai-org/sample-generator) models.

## Features

- A CLI for generating audio samples from the command line using Dance Diffusion models. (`generate.py`)
- A script for reducing the file size of Dance Diffusion models by removing data that is only needed for training and not inference. (`scripts/trim_model.py`)
- A discord bot for generating samples in Discord servers. (`start_discord_bot.py`)
    - [View the guide](https://github.com/sudosilico/sample-diffusion/blob/main/docs/discord.md)
- (Planned) A socket.io server that can be used as a Dance Diffusion service by applications written in any programming language that has a socket.io client. (`server.py`)

## Installation

### Requirements

- [git](https://git-scm.com/downloads) (to clone the repo)
- [conda](https://docs.conda.io/en/latest/) (to set up the python environment)

`conda` can be installed through [Anaconda](https://www.anaconda.com) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). To run on an Apple Silicon device, you will need to use a conda installation that includes Apple Silicon support, such as [Miniforge](https://github.com/conda-forge/miniforge).

### Cloning the repo

Clone the repo and `cd` into it:

```sh
git clone https://github.com/sudosilico/sample-diffusion
cd sample-diffusion
```

### Setting up the conda environment

Create the `conda` environment:

```sh
# If you're not running on an Apple Silicon machine:
conda env create -f environment.yml

# For Apple Silicon machines:
conda env create -f environment-mac.yml
```

This may take a few minutes as it will install all the necessary Python dependencies so that they will be available to the CLI script.

> Note: You must activate the `dd` conda environment after creating it. You can do this by running `conda activate dd` in the terminal. You will need to do this every time you open a new terminal window. Learn more about [conda environments.](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html)

```sh
conda activate dd
```

## Using the `generate.py` CLI

### Generating samples

Make a `models` folder and place your model in `models/model.ckpt`, then run the generator:

```sh
python generate.py
```

Alternatively, you can pass a custom model path as an argument:

```sh
python generate.py --ckpt models/some-other-model.ckpt
```

Your audio samples will then be in one of the following folders:

- `audio_out/generations/{seed}_{steps}` for generations (generate_unconditional)
- `audio_out/variations/{seed}_{steps}_{noise_level}` for variations (generate_variation)

along with a `meta.json` file containing the arguments, seed, and batch number.

### Using multiple batches

If you get an out-of-VRAM error, you may want to process in multiple batches.

Both of these commands will generate 25 audio samples in total, but the second one is split into multiple batches, allowing for a lower maximum VRAM usage.

```sh
# Generate 25 files, in 1 batch of 25 samples
python generate.py --samples 25

# Generate 25 files, in 5 batches of 5 samples
python generate.py --samples 5 --batches 5
```

When generating multiple batches, the first batch will use the passed seed (or a random seed if none was passed), and each subsequent batch will increment the seed by one.

### `generate.py` Command Line Arguments

| argument                   | type  | default             | desc                                               |
|----------------------------|-------|---------------------|----------------------------------------------------|
| --ckpt                     | str   | "models/model.ckpt" | path to the model to be used                       |
| --spc                      | int   | 65536               | the samples per chunk of the model                 |
| --sr                       | int   | 48000               | the samplerate of the model                        |
| --out_path                 | str   | "audio_out"         | path to the folder for the samples to be saved in  |
| --length_multiplier        | int   | 1                   | sample length multiplier for generate_variation           |
| --input_sr                 | int   | 44100               | samplerate of the input audio specified in --input |
| --noise_level              | float | 0.7                 | noise level for generate_variation                        |
| --steps                    | int   | 25                  | number of sampling steps                           |
| --samples                  | int   | 1                   | how many samples to generate per batch             |
| --batches                  | int   | 1                   | how many batches of samples to generate            |
| --seed                     | int   | -1                  | the seed (for reproducible sampling), -1 will be random every time.  |
| --input                    | str   | ""                  | path to the audio to be used for generate_variation. if missing or empty, generate_unconditional will be used.  |
| --remove_dc_offset         | flag  | False               | When this flag is set, a high pass filter will be applied to the input audio to remove DC offset. |
| --normalize                | flag  | False               | When this flag is set, output audio samples will be normalized. |
| --force_cpu                | flag  | False               | When this flag is set, processing will be done on the CPU. |
| --open                    | flag  | False                | When this flag is set, the containing folder will be opened once your samples are saved. |

## Using the model trimming script

`scripts/trim_model.py` can be used to reduce the file size of Dance Diffusion models by removing data that is only needed for training and not inference. For our first models, this reduced the model size by about 75% (from 3.46 GB to 0.87 GB).

To use it, simply pass the path to the model you want to trim as an argument:

```sh
python scripts/trim_model.py models/model.ckpt
```

This will create a new model file at `models/model_trim.ckpt`.
