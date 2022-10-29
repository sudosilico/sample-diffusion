import multiprocessing as mp
from multiprocessing.managers import SyncManager
import discord
from discord_bot.models_metadata import ModelsMetadata


class GenerationViewBase(discord.ui.View):
    def __init__(
        self,
        models_metadata: ModelsMetadata, 
        sync_manager: SyncManager, 
        request_queue: mp.Queue, 
        response_queue: mp.Queue,
        seed: int = -1,
        steps: int = 25,
        samples: int = 1,
        model_ckpt: str = None,
    ):
        super().__init__()

        self.interaction: discord.Interaction = None
        self.sync_manager = sync_manager
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.seed = seed
        self.steps = steps
        self.samples = samples
        self.model_ckpt = model_ckpt

    def add_button(
        self,
        callback,
        label: str,
        style: discord.ButtonStyle = discord.ButtonStyle.blurple,
        row: int = None,
        disabled: bool = False,
    ):
        button = ViewButton(
            callback, label=label, style=style, row=row, disabled=disabled
        )

        self.add_item(button)
        return button

    def add_modal_button(
        self,
        modal_factory,
        label: str,
        style: discord.ButtonStyle = discord.ButtonStyle.blurple,
        row: int = None,
        disabled: bool = False,
    ):
        # noise level button
        async def btn_callback(
            button: discord.ui.Button, interaction: discord.Interaction
        ):
            await interaction.response.send_modal(modal_factory(self))

        button = ViewButton(
            btn_callback, label=label, style=style, row=row, disabled=disabled
        )
        self.add_item(button)

        return button

    def get_embed(self):
        raise NotImplementedError


class ModelDropdown(discord.ui.Select):
    def __init__(
        self,
        models_metadata: ModelsMetadata, 
        parent_view: GenerationViewBase,
        placeholder: str = "Choose a model...",
    ):
        self.parent_view = parent_view

        options = []

        index = 1
        for model in models_metadata.meta_json["models"]:
            options.append(
                discord.SelectOption(
                    label=model["name"],
                    description=model["description"],
                    value=model["path"],
                )
            )
            index = index + 1

        super().__init__(
            placeholder=placeholder,
            min_values=1,
            max_values=1,
            options=options,
        )

    async def callback(self, interaction: discord.Interaction):
        self.parent_view.model_ckpt = self.values[0]
        self.placeholder = self.parent_view.model_ckpt

        await interaction.response.edit_message(
            embed=self.parent_view.get_embed(), view=self.parent_view
        )


class ViewButton(discord.ui.Button):
    def __init__(self, callback, *args, **kvargs):
        self.cb = callback

        super().__init__(*args, **kvargs)

    async def callback(self, interaction: discord.Interaction):
        if self.cb is not None:
            await self.cb(self, interaction)
