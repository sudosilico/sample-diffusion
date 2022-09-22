# Generator Guide

## Requirements

- git (to clone the repo)
- conda (to set up the python environment)

You can check for these commands with `git --version` and `conda --version`.

An easy way to get `conda` on Windows is by installing [Anaconda](https://www.anaconda.com/) and then using the "Anaconda Prompt".

## Clone the repo

Clone the repo and `cd` into it:

```
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
python ./generate.py
```

Alternatively, you can pass a custom model path as an argument:

```sh
python ./generate.py --ckpt models/some-other-model.ckpt
```

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
| --n_samples                | int   | 1                   | how many samples to produce / batch size           |
| --seed                     | int   | -1                  | the seed (for reproducible sampling), -1 will be random every time.  |
| --input                    | str   | ''                  | path to the audio to be used for audio2audio       |
