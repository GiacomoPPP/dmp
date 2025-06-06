from lightning import LightningModule
from numpy.typing import NDArray
import torch
import torch.nn as nn
import geotorch
from DmiConfig import DmiConfig
import math
import numpy as np


def compute_ecc(nh, index, lin):
    out = torch.zeros(
            size=(
                len(lin),
                index.max() + 1,
                nh.size()[1]
            ),
            device = lin.device
        )
    ecc = torch.nn.functional.sigmoid(50 * torch.sub(lin, nh))
    return torch.index_add(out, 1, index, ecc).movedim(0, 1)


def compute_ect_points(data, direction_list, threshold_list):
    nh = data.x @ direction_list
    return compute_ecc(nh, data.batch, threshold_list, data.num_graphs)


def compute_ect_edges(data, direction_list, threshold_list):
    nh = data.x @ direction_list
    eh, _ = nh[data.edge_index].max(dim=0)
    return compute_ecc(nh, data.batch, threshold_list) - compute_ecc(
        eh, data.batch[data.edge_index[0]], threshold_list
    )


def compute_ect_faces(data, direction_list, threshold_list):
    nh = data.x @ direction_list
    eh, _ = nh[data.edge_index].max(dim=0)
    fh, _ = nh[data.face].max(dim=0)
    return (
        compute_ecc(nh, data.batch, threshold_list)
        - compute_ecc(eh, data.batch[data.edge_index[0]], threshold_list)
        + compute_ecc(fh, data.batch[data.face[0]], threshold_list)
    )


class EctLayer(LightningModule):
    """docstring for EctLayer."""

    def __init__(self, config: DmiConfig, fixed=False):
        super().__init__()
        self.fixed = fixed
        self.threshold_list = (
            torch.linspace(-1, 1, config.bump_steps)
            .view(-1, 1, 1)
            .to(config.device)
        )
        if self.fixed:
            self.direction_list = torch.vstack(
                [
                    torch.sin(torch.linspace(0, 2 * torch.pi, config.num_thetas)),
                    torch.cos(torch.linspace(0, 2 * torch.pi, config.num_thetas)),
                ]
            ).to(config.device)
        else:
            self.direction_list = torch.Tensor(self.spherical_spiral(config.num_thetas)).to(config.device)
            self.direction_list /= self.direction_list.pow(2).sum(axis=1).sqrt().unsqueeze(1)
            self.direction_list = nn.Parameter(self.direction_list.T)

        if config.ect_type == "points":
            self.compute_ect = compute_ect_points
        elif config.ect_type == "edges":
            self.compute_ect = compute_ect_edges
        elif config.ect_type == "faces":
            self.compute_ect = compute_ect_faces


    def postinit(self):
        if not self.fixed:
            geotorch.constraints.sphere(self, "v")


    def forward(self, data):
        return self.compute_ect(data, self.direction_list, self.threshold_list)


    def spherical_spiral(self, points: int) -> NDArray:
        point_list = []
        diff_angle = np.pi * 2 / (math.sqrt(points))

        for i in range(points):
            theta = diff_angle * i
            z = 2*i / (points - 1) - 1
            radius =  np.sqrt(1 - z**2)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            point_list.append((x, y, z))

        return np.array(point_list)
