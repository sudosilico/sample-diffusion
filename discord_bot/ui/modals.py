import discord


class NoiseLevelModal(discord.ui.Modal):
    def __init__(self, selector):
        self.selector = selector

        super().__init__(
          title="Set noise level...",
        )

        self.input_text = discord.ui.InputText(
          label="Noise Level", 
          value=str(self.selector.noise_level), 
          style=discord.InputTextStyle.short
        )
        self.add_item(self.input_text)

    def is_valid(self, val):
        try:
            as_float = float(val)
            return as_float >= 0.0 and as_float <= 1.0
        except ValueError:
            return False

    async def callback(self, interaction: discord.Interaction):
        noise_level_str = self.input_text.value
        if not self.is_valid(noise_level_str):
          await interaction.response.send_message(
            "Error: Noise level must be a valid number within the range [0.0, 1.0].", 
            ephemeral=True
          )
        else:
          await interaction.response.defer()

          self.selector.noise_level = float(noise_level_str)

          await self.selector.interaction.edit_original_response(
            embed=self.selector.get_embed()
          )

          self.stop()


class StepsModal(discord.ui.Modal):
    def __init__(self, selector):
        self.selector = selector

        super().__init__(
          title="Set steps...",
        )

        self.input_text = discord.ui.InputText(
          label="Steps", 
          value=str(self.selector.steps), 
          style=discord.InputTextStyle.short
        )
        self.add_item(self.input_text)

    def is_valid(self, val):
        try:
            as_int = int(val)
            return as_int >= 1 and as_int <= 1000
        except ValueError:
            return False

    async def callback(self, interaction: discord.Interaction):
        steps_str = self.input_text.value
        if not self.is_valid(steps_str):
          await interaction.response.send_message(
            "Error: Steps must be a valid integer number within the range [0, 1000].", 
            ephemeral=True
          )
        else:
          await interaction.response.defer()

          self.selector.steps = int(steps_str)

          await self.selector.interaction.edit_original_response(
            embed=self.selector.get_embed()
          )

          self.stop()



class SamplesModal(discord.ui.Modal):
    def __init__(self, selector):
        self.selector = selector

        super().__init__(
          title="Set samples...",
        )

        self.input_text = discord.ui.InputText(
          label="Samples", 
          value=str(self.selector.samples), 
          style=discord.InputTextStyle.short
        )
        self.add_item(self.input_text)

    def is_valid(self, val):
        try:
            as_int = int(val)
            return as_int >= 1 and as_int <= 10
        except ValueError:
            return False

    async def callback(self, interaction: discord.Interaction):
        samples_str = self.input_text.value
        if not self.is_valid(samples_str):
          await interaction.response.send_message(
            "Error: Samples must be a valid integer number within the range [0, 1000].", 
            ephemeral=True
          )
        else:
          await interaction.response.defer()

          self.selector.samples = int(samples_str)
          
          await self.selector.interaction.edit_original_response(
            embed=self.selector.get_embed()
          )

          self.stop()


class SeedModal(discord.ui.Modal):
    def __init__(self, selector):
        self.selector = selector

        super().__init__(
          title="Set a seed...",
        )

        self.input_text = discord.ui.InputText(
          label="Seed", 
          placeholder="Set an integer seed...",
          style=discord.InputTextStyle.short
        )
        self.add_item(self.input_text)

    def is_valid(self, val):
        try:
            int(val)
            return True
        except ValueError:
            return False

    async def callback(self, interaction: discord.Interaction):
        seed_str = self.input_text.value
        if not self.is_valid(seed_str):
          await interaction.response.send_message(
            "Error: Seed must be a valid integer number.", 
            ephemeral=True
          )
        else:
          await interaction.response.defer()

          self.selector.seed = int(seed_str)
          
          await self.selector.interaction.edit_original_response(
            embed=self.selector.get_embed()
          )

          self.stop()
