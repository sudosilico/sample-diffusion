import discord

class BaseModal(discord.ui.Modal):
    def __init__(
        self,
        selector,
        title,
        label,
        placeholder,
        value=None,
        style=discord.InputTextStyle.short,
        validate=None,
        validation_fail_message=None,
        callback=None,
    ):
        self.selector = selector
        self.validate = validate
        self.validation_fail_message = validation_fail_message
        self.callback_ = callback

        super().__init__(
            title=title,
        )

        self.input_text = discord.ui.InputText(
            label=label, 
            value=str(value) if value is not None else None,
            placeholder=placeholder if placeholder is not None else None,
            style=style,
        )

        self.add_item(self.input_text)

    def is_valid(self, val):
        try:
            if self.validate is not None:
                return self.validate(val)
            else:
                return True
        except:
            return False

    async def callback(self, interaction: discord.Interaction):
        value_str = self.input_text.value
        if not self.is_valid(value_str):
            await interaction.response.send_message(self.validation_fail_message, ephemeral=True)
        else:
            await interaction.response.defer()

            if self.callback_ is not None:
              self.callback_(value_str)

            await self.selector.interaction.edit_original_response(
                embed=self.selector.get_embed(),
                view=self.selector
            )

            self.stop()


def create_noise_level_modal(selector):
    return BaseModal(
        selector=selector,
        title="Noise Level",
        label="Noise Level",
        placeholder="0.0 - 1.0",
        value=str(selector.noise_level),
        validate=lambda val: 0.0 <= float(val) <= 1.0,
        validation_fail_message="Error: Noise level must be a valid number within the range [0.0, 1.0].",
        callback=lambda val: setattr(selector, "noise_level", float(val)),
    )


def create_steps_modal(selector):
    return BaseModal(
        selector=selector,
        title="Steps",
        label="Steps",
        value=str(selector.steps),
        validate=lambda val: 1 <= int(val) <= 1000,
        validation_fail_message="Error: Steps must be a valid integer number within the range [1, 1000].",
        callback=lambda val: setattr(selector, "steps", int(val)),
    )


def create_samples_modal(selector):
    return BaseModal(
        selector=selector,
        title="Set samples...",
        label="Samples",
        value=str(selector.samples),
        validate=lambda val: 1 <= int(val) <= 1000,
        validation_fail_message="Error: Steps must be a valid integer number within the range [1, 1000].",
        callback=lambda val: setattr(selector, "samples", int(val)),
    )


def create_seed_modal(selector):
    def validate(val):
        try:
            int(val)
            return True
        except ValueError:
            return False

    def callback(val):
        selector.seed = val
        selector.set_random_seed_btn.disabled = False

    return BaseModal(
        selector=selector,
        title="Set a seed...",
        placeholder="Set an integer seed...",
        label="Seed",
        validate=validate,
        validation_fail_message="Error: Seed must be a valid integer number.",
        value=selector.seed,
        callback=lambda val: setattr(selector, "seed", int(val)),
    )
