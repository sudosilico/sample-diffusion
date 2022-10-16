import os
import discord
from discord_bot.models_metadata import ModelsMetadata
from discord_bot.ui.modals import create_noise_level_modal, create_samples_modal, create_seed_modal, create_steps_modal


def log_data(data):
    for i, v in enumerate(data):
        print("")
        print(f"i: {i}, {type(i)}")
        print(f"v: {v}, {type(v)}")
        print("")


class ModelDropdown(discord.ui.Select):
    def __init__(self, models_metadata: ModelsMetadata, selector):
        self.selector = selector

        options = []

        index = 1
        for model in models_metadata.meta_json["models"]:
            options.append(discord.SelectOption(
                label=model["name"],
                description=model["description"],
                value=model["path"]
            ))
            index = index + 1


        super().__init__(
            placeholder="Choose a model...",
            min_values=1,
            max_values=1,
            options=options,
        )

    async def callback(self, interaction: discord.Interaction):
        self.selector.model_name = self.values[0]
        self.placeholder = self.selector.model_name

        await interaction.response.edit_message(
            embed=self.selector.get_embed(),
            view=self.selector
        )


class ViewButton(discord.ui.Button):
    def __init__(self, callback, *args, **kvargs):
        self.cb = callback

        super().__init__(
            *args,
            **kvargs
        )

    async def callback(self, interaction: discord.Interaction):
        if self.cb is not None:
            await self.cb(self, interaction)


class GenerateVariationUIView(discord.ui.View):
    def __init__(self, models_metadata, file_path):
        self.interaction: discord.interactions.Interaction = None
        
        super().__init__()

        self.file_path = file_path
        self.seed = "Random"
        self.noise_level = 0.7
        self.steps = 25
        self.samples = 1
        self.model_ckpt = None
        self.model_name = None

        self.model_dropdown = ModelDropdown(models_metadata, self)
        self.add_item(self.model_dropdown)

        # noise level
        async def set_noise_level(
            button: discord.ui.Button, interaction: discord.Interaction
        ):
            await interaction.response.send_modal(create_noise_level_modal(self))

        self.set_noise_level_btn = ViewButton(set_noise_level, label="Set noise level", style=discord.ButtonStyle.blurple, row=2)
        self.add_item(self.set_noise_level_btn)

        # steps
        async def set_steps(
            button: discord.ui.Button, interaction: discord.Interaction
        ):
            await interaction.response.send_modal(create_steps_modal(self))

        self.set_steps_btn = ViewButton(set_steps, label="Set steps", style=discord.ButtonStyle.blurple, row=2)
        self.add_item(self.set_steps_btn)

        # samples
        async def set_samples(
            button: discord.ui.Button, interaction: discord.Interaction
        ):
            await interaction.response.send_modal(create_samples_modal(self))

        self.set_samples_btn = ViewButton(set_samples, label="Set samples", style=discord.ButtonStyle.blurple, row=2)
        self.add_item(self.set_samples_btn)

        # nonrandom seed
        async def set_nonrandom_seed(
            button: discord.ui.Button, interaction: discord.Interaction
        ):
            await interaction.response.send_modal(create_seed_modal(self))

        self.set_seed_btn = ViewButton(set_nonrandom_seed, label="Set seed", style=discord.ButtonStyle.gray, row=3)
        self.add_item(self.set_seed_btn)
        
        # random seed
        async def set_random_seed(
            button: discord.ui.Button, interaction: discord.Interaction
        ):
            self.seed = "Random"
            button.disabled = True

            # update embed
            await interaction.response.edit_message(
                embed=self.get_embed(),
                view=self
            )

        self.set_random_seed_btn = ViewButton(set_random_seed, label="Use random seed", style=discord.ButtonStyle.gray, row=3)
        self.set_random_seed_btn.disabled = True
        self.add_item(self.set_random_seed_btn)


    @discord.ui.button(label="Generate", style=discord.ButtonStyle.green, row=4)
    async def generate_clicked(
      self, button: discord.ui.Button, interaction: discord.Interaction
    ):
        print(f"Variation completed. Deleting '{self.file_path}'")

        if os.path.exists(self.file_path):
            os.remove(self.file_path)
            print("Deleted")
        else:
            print("The file does not exist")

        await interaction.response.edit_message(
            embed=None,
            view=None,
            content="Generating..."
        )

    def get_embed(self):
        harmonai_blue = 0x01239b
        embed = discord.Embed(title="Variation options:", color=harmonai_blue)

        embed.add_field(name="Noise Level", value=self.noise_level)
        embed.add_field(name="Steps", value=self.steps)
        embed.add_field(name="Samples", value=self.samples, inline=True)
        
        embed.add_field(name="Seed", value=self.seed)
        embed.add_field(name="Model", value=self.model_name)

        return embed

    async def on_timeout(self):
        print(f"Timed out. Deleting '{self.file_path}'")

        if os.path.exists(self.file_path):
            os.remove(self.file_path)
            print("Deleted")
        else:
            print("The file does not exist")

        await self.interaction.edit_original_response(
            view=None,
            embed=None,
            content="Timed out..."
        )

