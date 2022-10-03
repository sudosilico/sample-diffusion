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

## CLI Options

By default, models are expected to be in a `models` folder in the project root. You can can set a custom models folder using the `--models_path` argument.

The bot will create a `models.json` file in the models path, containing an entry for each `.ckpt` file. This is where you can set a custom name, description, sample_rate, and chunk_size for each individual model file.

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

By default, the maximum size of the request queue is 10. After this, users will be told the queue is full instead of their requests being added. You can set a custom maximum using the `--max_queue_size` argument.

## Starting the bot

To start the bot, run the following command from the [conda environment](https://github.com/sudosilico/sample-diffusion#installation):

```sh
python start_discord_bot.py
```

## `start_discord_bot.py` Command Line Arguments

| argument | type | default | desc |
| --- | --- | --- | --- |
| --models_path | str | "models" | Path to the folder containing your models |
| --output_path | str | "outputs_from_discord_bot" | Path to the folder where generated audio will be saved |
| --max_queue_size | int | 10 | Maximum number of requests that can be queued at once |