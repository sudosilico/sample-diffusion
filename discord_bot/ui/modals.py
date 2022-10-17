import discord


class BaseModal(discord.ui.Modal):
    def __init__(
        self,
        parent_view,
        title,
        label,
        placeholder=None,
        value=None,
        style=discord.InputTextStyle.short,
        validate=None,
        validation_fail_message=None,
        callback=None,
    ):
        self.parent_view = parent_view
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

            await self.parent_view.interaction.edit_original_response(
                embed=self.parent_view.get_embed(),
                view=self.parent_view
            )

            self.stop()


def create_noise_level_modal(parent_view):
    return BaseModal(
        parent_view=parent_view,
        title="Noise Level",
        label="Noise Level",
        placeholder="0.0 - 1.0",
        value=str(parent_view.noise_level),
        validate=lambda val: 0.0 <= float(val) <= 1.0,
        validation_fail_message="Error: Noise level must be a valid number within the range [0.0, 1.0].",
        callback=lambda val: setattr(parent_view, "noise_level", float(val)),
    )


def create_steps_modal(parent_view):
    return BaseModal(
        parent_view=parent_view,
        title="Steps",
        label="Steps",
        value=str(parent_view.steps),
        validate=lambda val: 1 <= int(val) <= 1000,
        validation_fail_message="Error: Steps must be a valid integer number within the range [1, 1000].",
        callback=lambda val: setattr(parent_view, "steps", int(val)),
    )


def create_samples_modal(parent_view):
    return BaseModal(
        parent_view=parent_view,
        title="Set samples...",
        label="Samples",
        value=str(parent_view.samples),
        validate=lambda val: 1 <= int(val) <= 1000,
        validation_fail_message="Error: Samples must be a valid integer number within the range [1, 1000].",
        callback=lambda val: setattr(parent_view, "samples", int(val)),
    )


def create_length_multiplier_modal(parent_view):
    return BaseModal(
        parent_view=parent_view,
        title="Set length multiplier...",
        label="Length Multiplier",
        value=str(parent_view.length_multiplier),
        validate=lambda val: (1 <= int(val) <= 1000) or (int(val) == -1),
        validation_fail_message="Error: Length multiplier must be a valid integer number within the range [1, 1000], or -1.",
        callback=lambda val: setattr(parent_view, "length_multiplier", int(val)),
    )


def create_seed_modal(parent_view):
    def validate(val):
        try:
            int(val)
            return True
        except ValueError:
            return False

    def callback(val):
        parent_view.seed = int(val)
        parent_view.set_random_seed_btn.disabled = False

    return BaseModal(
        parent_view=parent_view,
        title="Set a seed...",
        placeholder="Set an integer seed...",
        label="Seed",
        validate=validate,
        validation_fail_message="Error: Seed must be a valid integer number.",
        value=parent_view.seed,
        callback=callback,
    )
