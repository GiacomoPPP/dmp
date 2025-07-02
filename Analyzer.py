from collections import OrderedDict

import sys

from pathlib import Path

import torch

from DmiConfig import DmiConfig

config = DmiConfig()


class Analyzer:

    saved_models_path = "saved_models"

    def _load_model(self) -> tuple[OrderedDict, str]:
        chosen_model_path, chosen_model_name = self._ask_which_model()

        return torch.load(chosen_model_path), chosen_model_name


    def _ask_which_model(self) -> tuple[Path, str]:

        files: list[Path] = self._get_saved_model_list()

        if config.fast_run:
            choice = 0
        else:
            for i, file in enumerate(files, start=1):
                print(f"{i}: {file.name}")
            choice = int(input("Select a file by number: ")) - 1

            for _ in range(len(files) + 1):
                sys.stdout.write("\033[F\033[K")
            sys.stdout.flush()

        chosen_model_path: Path = files[choice]

        chosen_file_name: str = files[choice].name.removesuffix(".pth")

        return chosen_model_path, chosen_file_name


    def _get_saved_model_list(self) -> list[Path]:
        directory = Path(self.saved_models_path).resolve()

        file_list = sorted(directory.iterdir(), key=lambda f: f.stat().st_ctime, reverse=True)
        file_list = [f for f in file_list if f.is_file()]

        return file_list