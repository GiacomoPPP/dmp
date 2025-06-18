
import torch
from torch import Tensor

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import numpy as np

from Analyzer import Analyzer


class DirectionListAnalyzer(Analyzer):

    def __call__(self):

        model, model_name = self._load_model()

        direction_list: Tensor = self._get_direction_list(model)

        self._analyze_average_sequential_distance(direction_list, model_name)

        self._plot_direction_list(direction_list)


    def _get_direction_list(self, model) -> Tensor:
        direction_list: Tensor = model["ectlayer.direction_list"]
        direction_list = torch.nn.functional.normalize(direction_list, p=2, dim = 0)

        return direction_list


    def _analyze_average_sequential_distance(self, direction_list: Tensor, model_name: str) -> None:

        difference_list = direction_list[:, 1:] - direction_list[:, :-1]

        total_sequential_norm: float = difference_list.norm(dim=0).sum()

        average_norm: float = total_sequential_norm / direction_list.shape[1]

        print(f"Average norm for model {model_name}: {average_norm}")


    def _plot_direction_list(self, direction_list: Tensor) -> None:
        direction_list = direction_list.T
        direction_list = direction_list.tolist()

        ax = plt.axes(projection='3d')

        x_list, y_list, z_list = zip(*direction_list)

        #ax.scatter3D(x_list, y_list, z_list)
        ax.plot3D(x_list, y_list, z_list)

        n = len(x_list)

        colors = cm.viridis(np.linspace(0, 1, n - 1))

        for i in range(n - 1):
            ax.plot([x_list[i], x_list[i+1]],
                    [y_list[i], y_list[i+1]],
                    [z_list[i], z_list[i+1]],
                    color=colors[i])

        for i, (x, y, z) in enumerate(zip(x_list, y_list, z_list)):
            ax.text(x, y, z, str(i), fontsize=8)

        plt.show()