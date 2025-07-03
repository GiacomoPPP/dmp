from typing import Any
import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection


from torch_geometric.data import Data as Graph

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
            ax.plot(x, y, z, color='black')

        self._add_axes_labels(ax)

        self._add_limits(ax)

        self._colorize(ax, coordinate_list, G)

        plt.show()

    def _get_coordinate_list(self, graph: Graph) -> dict[int, Any]:
        pos_np = graph.x.numpy()
        coordinate_list = {i: pos_np[i] for i in range(len(pos_np))}
        return coordinate_list


    def _add_axes_labels(self, ax) -> None:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')


    def _add_limits(self, ax) -> None:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

    def _colorize(self, ax, coordinate_list, G) -> None:
        segments = []
        colors = []

        for edge in G.edges():
            p1 = coordinate_list[edge[0]]
            p2 = coordinate_list[edge[1]]
            segments.append([p1, p2])
            avg_z = (p1[2] + p2[2]) / 2  # example scalar value
            colors.append(avg_z)

        lc = Line3DCollection(segments, cmap='viridis', linewidths=2)
        lc.set_array(np.array(colors))
        ax.add_collection3d(lc)