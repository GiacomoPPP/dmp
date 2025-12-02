from typing import Any

import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import matplotlib as mpl

from torch_geometric.data import Data as Graph

from Analyzer import DmpConfig

mpl.rcParams['figure.figsize'] = 4, 4

class GraphDrawer:
    def __call__(self, graph: Graph):

        graph = graph.cpu()

        G = to_networkx(graph, to_undirected=True)

        coordinate_list: dict = self._get_coordinate_list(graph)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for edge in G.edges():
            x = [coordinate_list[edge[0]][0], coordinate_list[edge[1]][0]]
            y = [coordinate_list[edge[0]][1], coordinate_list[edge[1]][1]]
            z = [coordinate_list[edge[0]][2], coordinate_list[edge[1]][2]]
            ax.plot(x, y, z, linewidth = 0.01)

        self._add_axes_labels(ax)

        self._add_limits_and_ticks(ax)

        self._hide_ticks(ax)

        ax.view_init(azim=10,elev=20)

        self._colorize(ax, coordinate_list, G)

        ax.set_aspect('equal')

        plt.savefig("pdf/graph.pdf", pad_inches=0, bbox_inches='tight')

        plt.show()


    def _get_coordinate_list(self, graph: Graph) -> dict[int, Any]:
        pos_np = graph.x.numpy()
        coordinate_list = {i: pos_np[i] for i in range(len(pos_np))}
        return coordinate_list


    def _add_axes_labels(self, ax) -> None:
        ax.set_xlabel('x', labelpad=-12)
        ax.set_ylabel('y', labelpad=-12)
        ax.set_zlabel('z', labelpad=-15)


    def _add_limits_and_ticks(self, ax) -> None:
        x_range = 1
        y_range = 0.7
        z_range = 0.4
        ax.set_xlim(-x_range, x_range)
        ax.set_ylim(-y_range, y_range)
        ax.set_zlim(-z_range, z_range)
        ax.xaxis.set_ticks(np.arange(-x_range, x_range, 0.2))
        ax.yaxis.set_ticks(np.arange(-y_range, y_range, 0.2))
        ax.zaxis.set_ticks(np.arange(-z_range, z_range, 0.2))

    def _hide_ticks(self, ax) -> None:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])


    def _colorize(self, ax, coordinate_list, G) -> None:
        segments = []
        colors = []

        for edge in G.edges():
            p1 = coordinate_list[edge[0]]
            p2 = coordinate_list[edge[1]]
            segments.append([p1, p2])
            avg_z = (p1[2] + p2[2]) / 2
            colors.append(avg_z)

        lc = Line3DCollection(segments, cmap='viridis', linewidths=1.4)
        lc.set_array(np.array(colors))
        ax.add_collection3d(lc)

if __name__ == "__main__":

    from DatasetGenerator import DatasetGenerator

    import MplStyle

    config = DmpConfig()

    datasetGenerator : DatasetGenerator= DatasetGenerator()

    graph_list, _ = datasetGenerator.get_dataset(config.dataset)

    maxindex = int(np.argmax([len(graph.x) for graph in graph_list]))

    from GraphDrawer import GraphDrawer

    graphDrawer = GraphDrawer()

    graphDrawer(graph_list[maxindex])