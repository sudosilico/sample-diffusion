# Using the `sample-diffusion` Discord bot

## Requirements

You must have the project and conda environment installed and activated. View the [installation instructions](https://github.com/sudosilico/sample-diffusion#installation) in the main README to learn more.

## Setting up your Discord bot token

Create a [Discord Application](https://discord.com/developers/applications/) for your bot. Click on 'Bot', and create a bot profile, then click on OAuth2 -> URL Generator, create an invite link with the 'bot' scope and the permissions listed below.

- `Send Messages`
- `Embed Links`
- `Attach Files`
- `Add Reactions`
- `Use Slash Commands`

Then, navigate to the generated link and add the bot to your server.

In the repository root, create a `.env` file with the following contents:

```
DISCORD_BOT_TOKEN=<your bot token>
```

You can find your bot token in the 'Bot' section of your Discord Application, under your bot's username field.

## Adding your models

By default, models are expected to be in a `models` folder in the project root. You can can set a custom models folder using the `models_path` CLI argument.

The bot will create a `models.json` file in the models path, containing an entry for each `.ckpt` file in the models path. This is where you can set a custom name, description, sample_rate, and chunk_size for each individual model file.

An example `models.json` can be seen here:

```json
{
    "models": [
        {
            "path": "glitch-440k.ckpt",
            "name": "glitch",
            "description": "Trained on clips from samples provided by glitch.cool",
            "sample_rate": 48000,
            "chunk_size": 65536
        },
        {
            "path": "jmann-large-580k.ckpt",
            "name": "jmann large",
            "description": "Trained on clips from Jonathan Mann's Song-A-Day project",
            "sample_rate": 48000,
            "chunk_size": 131072
        }
    ]
}
```

By default, generated audio is saved in a `outputs_from_discord_bot` folder. You can set a custom output path using the `--output_path` argument.

## Starting the bot

To start the bot, run the following command from the [conda environment](https://github.com/sudosilico/sample-diffusion#installation):

```sh
python start_discord_bot.py
```

## Using the bot

The bot provides the following slash command:

- **/generate** - Generates a number of audio samples using one of the available models. Arguments:
    - `model` - The model to use for sample generation _(required)_
    - `samples`- The number of samples to generate (default: `1`)
    - `steps` - The number of steps to perform during generation (default: `25`)
    - `seed` - The random seed, or `-1` for a random one (default: `-1`)

## `start_discord_bot.py` Command Line Arguments

| argument | type | default | desc |
| --- | --- | --- | --- |
| --models_path | str | "models" | Path to the folder containing your models |
| --output_path | str | "outputs_from_discord_bot" | Path to the folder where generated audio will be saved |
| --config_path | str | "bot_config.ini" | Path to the config file |
| --open | flag | False | When this flag is used, the bot will open the output folder after generating |

## Bot configuration

Here is an example `bot_config.ini` file that may be used:

```ini
[DEFAULT]
max_queue_size = 10
max_samples = 3
max_steps = 250

[admin]
max_queue_size = 500
max_samples = 10
max_steps = 500
```

You can also create a category for individual discord users, using the `[user:<discord id>]` form:

```ini
[user:123456578987654321]
max_queue_size = 20
```

If a category does not contain a certain config value, the value under `[DEFAULT]` will be used. This also applies to user-specific configs; `[DEFAULT]` will be used as a fallback for `[user:id]` over `[admin]` even when that user is an admin. 