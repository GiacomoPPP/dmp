import sys

from collections import OrderedDict

import torch
from torch import Tensor

import matplotlib.pyplot as plt

from numpy import linalg

from pathlib import Path

class DirectionListAnalyzer:

    saved_models_path = "saved_models"

    def __call__(self):

        model, model_name = self._get_model()

        direction_list: Tensor = self._get_direction_list(model)

        self._analyze_average_sequential_distance(direction_list, model_name)

        self._plot_direction_list(direction_list)


    def _get_model(self) -> tuple[OrderedDict, str]:
        chosen_model_path, chosen_model_name = self._ask_which_model()

        return torch.load(chosen_model_path), chosen_model_name


    def _ask_which_model(self) -> tuple[Path, str]:

        files: list[Path] = self._get_saved_model_list()

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


    def _get_direction_list(self, model) -> Tensor:
        direction_list: Tensor = model["ectlayer.direction_list"]
        direction_list = torch.nn.functional.normalize(direction_list, p=2, dim = 0)

        return direction_list


    def _analyze_average_sequential_distance(self, direction_list: Tensor, model_name: str) -> None:

        total_sequential_norm: float = torch.sum((direction_list[1:] - direction_list[:-1])**2)

        average_norm: float = total_sequential_norm / direction_list.shape[1]

        print(f"Average norm for model {model_name}: {average_norm}")


    def _plot_direction_list(self, direction_list: Tensor) -> None:
        direction_list = direction_list.T
        direction_list = direction_list.tolist()

        ax = plt.axes(projection='3d')

        x_list, y_list, z_list = zip(*direction_list)

        ax.scatter3D(x_list, y_list, z_list)
        #ax.plot3D(x_list, y_list, z_list)

        for i, (x, y, z) in enumerate(zip(x_list, y_list, z_list)):
            ax.text(x, y, z, str(i), fontsize=8)

        plt.show()