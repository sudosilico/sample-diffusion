import json
import os
import discord


class ModelsMetadata:
    def __init__(self, models_path: str):
        if not os.path.exists(models_path):
            os.makedirs(models_path)

        meta_path = os.path.join(models_path, "models.json")
        meta_json = None

        # get ckpt files from models path
        ckpt_paths = []
        for root, dirs, files in os.walk(models_path):
            for file in files:
                if file.endswith(".ckpt"):
                    ckpt_paths.append(file)

        # load metadata from json file
        if not os.path.exists(meta_path):
            meta_json = {"models": []}
        else:
            with open(meta_path, "r") as f:
                meta_json = json.load(f)
        pass

        # ensure each checkpoint file has a metadata entry
        for ckpt in ckpt_paths:
            has_meta = False

            for model in meta_json["models"]:
                if model["path"] == ckpt:
                    has_meta = True
                    break

            if not has_meta:
                meta_json["models"].append(
                    {
                        "name": os.path.splitext(ckpt)[0],
                        "description": "",
                        "path": ckpt,
                        "sample_rate": 48000,
                        "chunk_size": 65536,
                    }
                )

        # save updated metadata to json file
        with open(meta_path, "w") as f:
            json.dump(meta_json, f, indent=4)

        self.ckpt_paths = ckpt_paths
        self.meta_json = meta_json

    def get_meta(self, ckpt: str):
        for model in self.meta_json["models"]:
            if model["path"] == ckpt:
                return model

    def get_ckpt_paths(self, ctx: discord.commands.context.AutocompleteContext):
        return self.ckpt_paths
