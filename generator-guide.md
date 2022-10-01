# Guide to the `generate.py` CLI

## Requirements

- [git](https://git-scm.com/downloads) (to clone the repo)
- [conda](https://docs.conda.io/en/latest/) (to set up the python environment)

`conda` can be installed through [Anaconda](https://www.anaconda.com) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

## Clone the repo

Clone the repo and `cd` into it:

```sh
git clone https://github.com/sudosilico/sample-diffusion
cd sample-diffusion
```

## Set up conda environment

Create the `conda` environment:

```sh
conda env create -f environment.yml
```

This may take a few minutes as it will install all the necessary Python dependencies so that they will be available to the CLI script.

While you only need to _create_ the `conda` environment once, you'll need to _activate_ it after creating it and once for each new terminal session so that python can access the dependencies.

```sh
conda activate sd_backend
```

## Running the Generator

Make a `models` folder and place your model in `models/model.ckpt`, then run the generator:

```sh
python generate.py
```

Alternatively, you can pass a custom model path as an argument:

```sh
python generate.py --ckpt models/some-other-model.ckpt
```

Your audio samples will then be in one of the following folders:

`audio_out/generations/{seed}_{n_steps}` for generations (noise2audio)
`audio_out/variations/{seed}_{n_steps}_{noise_level}` for variations (audio2audio)

along with a `meta.json` file containing the passed arguments, seed, and batch number.

## Multiple Batches

If you get an out-of-VRAM error, you may want to process in multiple batches. 

Both of these commands will generate 25 audio samples in total, but the second one is split into multiple batches, allowing for a lower maximum VRAM usage.

```sh
# Generate 25 files, in 1 batch of 25 samples
python generate.py --n_samples 25

# Generate 25 files, in 5 batches of 5 samples
python generate.py --n_samples 5 --n_batches 5
```

When generating multiple batches, the first batch will use the passed seed (or a random seed if none was passed), and each subsequent batch will increment the seed by one.

## `generate.py` Arguments

| argument                   | type  | default             | desc                                               |
|----------------------------|-------|---------------------|----------------------------------------------------|
| --ckpt                     | str   | "models/model.ckpt" | path to the model to be used                       |
| --spc                      | int   | 65536               | the samples per chunk of the model                 |
| --sr                       | int   | 48000               | the samplerate of the model                        |
| --out_path                 | str   | "audio_out"         | path to the folder for the samples to be saved in  |
| --sample_length_multiplier | int   | 1                   | sample length multiplier for audio2audio           |
| --input_sr                 | int   | 44100               | samplerate of the input audio specified in --input |
| --noise_level              | float | 0.7                 | noise level for audio2audio                        |
| --n_steps                  | int   | 25                  | number of sampling steps                           |
| --n_samples                | int   | 1                   | how many samples to generate per batch             |
| --n_batches                | int   | 1                   | how many batches of samples to generate            |
| --seed                     | int   | -1                  | the seed (for reproducible sampling), -1 will be random every time.  |
| --input                    | str   | ""                | path to the audio to be used for audio2audio       |
