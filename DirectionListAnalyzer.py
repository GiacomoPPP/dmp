from mpl_toolkits.mplot3d import Axes3D
import torch
from torch import Tensor

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

import numpy as np

from Analyzer import Analyzer

mpl.rcParams['figure.figsize'] = 4, 4


class DirectionListAnalyzer(Analyzer):

    def __call__(self):

        while True:
            model, model_name = self._load_model()

            direction_list: Tensor = self._get_direction_list(model)

            self._analyze_average_sequential_distance(direction_list, model_name)

            self._plot_direction_list(direction_list, model_name)


    def _get_direction_list(self, model) -> Tensor:
        direction_list: Tensor = model["ectlayer.direction_list"]
        direction_list = torch.nn.functional.normalize(direction_list, p=2, dim = 0)

        return direction_list


    def _analyze_average_sequential_distance(self, direction_list: Tensor, model_name: str) -> None:

        difference_list = direction_list[:, 1:] - direction_list[:, :-1]

        total_sequential_norm: float = difference_list.norm(dim=0).sum()

        average_norm: float = total_sequential_norm / direction_list.shape[1]

        print(f"Average norm for model {model_name}: {average_norm}")


    def _plot_direction_list(self, direction_list: Tensor, model_name: str) -> None:
        direction_list = direction_list.T
        direction_list = direction_list.tolist()

        _ = plt.figure()

        ax: Axes3D = plt.axes(projection='3d')
        print(type(ax))

        plt.title(model_name)

        x_list, y_list, z_list = zip(*direction_list)

        self._add_points(x_list, y_list, z_list, ax)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        plt.savefig("pdf/direction_list.pdf", bbox_inches='tight', pad_inches=0)

        self._add_axes_labels(ax)

        plt.show()


    def _add_points(self, x_list: list, y_list: list, z_list: list, ax: Axes3D) -> None:
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



    def _add_axes_labels(self, ax: Axes3D) -> None:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')