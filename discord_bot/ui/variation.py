import os
import discord
import torch
from discord_bot.commands import DiffusionRequest
from discord_bot.ui.common import GenerationViewBase, ModelDropdown
from discord_bot.ui.modals import (
    create_length_multiplier_modal,
    create_noise_level_modal,
    create_samples_modal,
    create_seed_modal,
    create_steps_modal,
)
from discord_bot.response import handle_generation_response


class VariationUIView(GenerationViewBase):
    def __init__(
        self,
        models_metadata,
        file_path,
        file_name,
        sync_manager,
        request_queue,
        response_queue,
    ):
        super().__init__(
            models_metadata=models_metadata,
            sync_manager=sync_manager,
            request_queue=request_queue,
            response_queue=response_queue,
        )

        self.file_path = file_path
        self.file_name = file_name
        self.noise_level = 0.7
        self.length_multiplier = -1

        # model selector dropdown
        self.model_dropdown = ModelDropdown(models_metadata, self)
        self.add_item(self.model_dropdown)

        # noise level button
        self.set_noise_level_btn = self.add_modal_button(
            modal_factory=create_noise_level_modal, label="Set noise level", row=2
        )

        # steps button
        self.set_steps_btn = self.add_modal_button(
            modal_factory=create_steps_modal, label="Set steps", row=2
        )

        # samples button
        self.set_samples_btn = self.add_modal_button(
            modal_factory=create_samples_modal, label="Set samples", row=2
        )

        # length multiplier button
        self.set_length_multiplier_btn = self.add_modal_button(
            modal_factory=create_length_multiplier_modal,
            label="Set length multiplier",
            row=2,
        )

        # nonrandom seed button
        self.set_seed_btn = self.add_modal_button(
            modal_factory=create_seed_modal, label="Set seed", style=discord.ButtonStyle.gray, row=3
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
            row=3,
            disabled=True,
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

        if self.will_generate:
            await interaction.response.send_message(
                "Error: Already submitted this request.", ephemeral=True
            )
            return

        self.will_generate = True

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
            gen_type="variation",
            input=self.file_path,
            input_name=self.file_name,
            noise_level=self.noise_level,
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
            parent_view=self,
        )

    def get_embed(self):
        embed = discord.Embed(title="Variation options:", color=0x01239B)

        embed.add_field(name="Noise Level", value=self.noise_level)
        embed.add_field(name="Steps", value=self.steps)
        embed.add_field(name="Samples", value=self.samples)

        embed.add_field(
            name="Length Multiplier",
            value=self.length_multiplier
            if self.length_multiplier != -1
            else "Full Length",
        )
        embed.add_field(name="Seed", value="Random" if self.seed == -1 else self.seed)
        embed.add_field(name="Model", value=self.model_ckpt)

        return embed

    async def on_timeout(self):
        if self.will_generate:
            return
            
        print(f"Timed out. Deleting '{self.file_path}'")

        if os.path.exists(self.file_path):
            os.remove(self.file_path)
            print("  Deleted")
        else:
            print("  The file does not exist")

        await self.interaction.edit_original_response(
            view=None, embed=None, content="Timed out..."
        )
