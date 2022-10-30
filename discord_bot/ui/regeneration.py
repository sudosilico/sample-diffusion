from multiprocessing.managers import SyncManager
import discord
import torch
import multiprocessing as mp
from discord_bot.models_metadata import ModelsMetadata
from discord_bot.request import DiffusionRequest
from discord_bot.response import handle_generation_response
from discord_bot.ui.common import GenerationViewBase, ModelDropdown
from discord_bot.ui.modals import create_samples_modal, create_seed_modal, create_steps_modal

class RegenerationUIView(GenerationViewBase):
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
        super().__init__(
            models_metadata=models_metadata,
            sync_manager=sync_manager,
            request_queue=request_queue,
            response_queue=response_queue,
            seed=seed,
            steps=steps,
            samples=samples,
            model_ckpt=model_ckpt,
        )

        # model selector dropdown
        self.model_dropdown = ModelDropdown(models_metadata, self, placeholder=model_ckpt)
        self.add_item(self.model_dropdown)

        # steps button
        self.set_steps_btn = self.add_modal_button(
            modal_factory=create_steps_modal, label="Set steps", row=2
        )

        # samples button
        self.set_samples_btn = self.add_modal_button(
            modal_factory=create_samples_modal, label="Set samples", row=2
        )

        # nonrandom seed button
        self.set_seed_btn = self.add_modal_button(
            modal_factory=create_seed_modal, label="Set seed", style=discord.ButtonStyle.gray, row=2
        )

        # random seed button
        async def set_random_seed(
            button: discord.ui.Button, interaction: discord.Interaction
        ):
            self.seed = -1
            button.disabled = True

            # update embed
            await interaction.response.edit_message(embed=self.get_embed(), view=self)

        self.set_random_seed_btn = self.add_button(
            callback=set_random_seed,
            label="Use random seed",
            style=discord.ButtonStyle.gray,
            row=2,
            disabled=(seed == -1),
        )

    # generate button
    @discord.ui.button(label="Generate", style=discord.ButtonStyle.green, row=4)
    async def generate_clicked(
        self, button: discord.ui.Button, interaction: discord.Interaction
    ):
        if self.model_ckpt is None:
            await interaction.response.send_message(
                "Error: You must select a model.", ephemeral=True
            )
            return

        start_event = self.sync_manager.Event()
        done_event = self.sync_manager.Event()
        progress_queue = self.sync_manager.Queue()

        seed = self.seed
        seed = seed if seed != -1 else torch.seed()
        seed = seed % 4294967295

        request = DiffusionRequest(
            model=self.model_ckpt,
            seed=seed,
            samples=self.samples,
            steps=self.steps,
            start_event=start_event,
            done_event=done_event,
            progress_queue=progress_queue,
        )
        # request_queue_size = self.request_queue.qsize()
        self.request_queue.put(request)

        content = f"Your request has been queued."  # There are {request_queue_size} tasks ahead of yours in the queue."
        await self.interaction.edit_original_response(
            embed=self.get_embed(),
            view=None,
            content=content,
        )

        await handle_generation_response(
            ctx=None,
            start_event=request.start_event,
            done_event=request.done_event,
            progress_queue=request.progress_queue,
            request=request,
            response_queue=self.response_queue,
            interaction=self.interaction,
            request_embed=self.get_embed(),
        )

    def get_embed(self):
        embed = discord.Embed(title="Generation options:", color=0x01239b)

        embed.add_field(name="Steps", value=self.steps)
        embed.add_field(name="Samples", value=self.samples)
        
        embed.add_field(name="Seed", value="Random" if self.seed == -1 else self.seed)
        embed.add_field(name="Model", value=self.model_ckpt)

        return embed