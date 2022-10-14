import os
import discord
from discord_bot.models_metadata import ModelsMetadata
from discord_bot.ui.modals import NoiseLevelModal, SamplesModal, SeedModal, StepsModal


def log_data(data):
    for i, v in enumerate(data):
        print("")
        print(f"i: {i}, {type(i)}")
        print(f"v: {v}, {type(v)}")
        print("")


class ModelDropdown(discord.ui.Select):
    def __init__(self, models_metadata: ModelsMetadata):
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
            options=options
        )

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer()


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


class ModelSelectorView(discord.ui.View):
    interaction: discord.interactions.Interaction

    def __init__(self, models_metadata):
        super().__init__()

        self.seed = "Random"
        self.noise_level = 0.7
        self.steps = 25
        self.samples = 1

        self.model_dropdown = ModelDropdown(models_metadata)
        self.add_item(self.model_dropdown)

        # noise level
        async def set_noise_level(
            button: discord.ui.Button, interaction: discord.Interaction
        ):
            await interaction.response.send_modal(NoiseLevelModal(self))

        set_noise_level_btn = ViewButton(set_noise_level, label="Set noise level", style=discord.ButtonStyle.blurple, row=2)
        self.add_item(set_noise_level_btn)

        # steps
        async def set_steps(
            button: discord.ui.Button, interaction: discord.Interaction
        ):
            await interaction.response.send_modal(StepsModal(self))

        set_steps_btn = ViewButton(set_steps, label="Set steps", style=discord.ButtonStyle.blurple, row=2)
        self.add_item(set_steps_btn)

        # samples
        async def set_samples(
            button: discord.ui.Button, interaction: discord.Interaction
        ):
            await interaction.response.send_modal(SamplesModal(self))

        set_samples_btn = ViewButton(set_samples, label="Set samples", style=discord.ButtonStyle.blurple, row=2)
        self.add_item(set_samples_btn)

        # nonrandom seed
        async def set_nonrandom_seed(
            button: discord.ui.Button, interaction: discord.Interaction
        ):
            await interaction.response.send_modal(SeedModal(self))

        set_seed_btn = ViewButton(set_nonrandom_seed, label="Set seed", style=discord.ButtonStyle.gray, row=3)
        self.add_item(set_seed_btn)
        
        # random seed
        async def set_random_seed(
            button: discord.ui.Button, interaction: discord.Interaction
        ):
            await interaction.response.defer()
            self.seed = "Random"

            # update embed
            await self.interaction.edit_original_response(
                embed=self.get_embed()
            )

        set_random_seed_btn = ViewButton(set_random_seed, label="Use random seed", style=discord.ButtonStyle.gray, row=3)
        self.add_item(set_random_seed_btn)


    @discord.ui.button(label="Generate", style=discord.ButtonStyle.green, row=4)
    async def generate_clicked(
      self, button: discord.ui.Button, interaction: discord.Interaction
    ):
        values = self.model_dropdown.values
        print(f"values: {values}")

        await interaction.response.defer()


    def get_embed(self):
        harmonai_blue = 0x01239b
        embed = discord.Embed(title="Variation options:", color=harmonai_blue)

        embed.add_field(name="Noise Level", value=self.noise_level)
        embed.add_field(name="Steps", value=self.steps)
        embed.add_field(name="Samples", value=self.samples, inline=True)
        
        embed.add_field(name="Seed", value=self.seed, inline=True)

        return embed

