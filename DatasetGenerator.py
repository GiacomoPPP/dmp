import random

from typing import Literal

import numpy as np

import torch
from torch import Tensor

from torch_geometric.data import Data as Graph

from DatasetParser import DatasetParser
from DatasetSplitter import DatasetSplitter
from DmpConfig import DmpConfig
from DmpDataset import DmpDataset

config = DmpConfig()

class DatasetGenerator:


    def __init__(self):
        self.datasetParser = DatasetParser()


    def get_dataset_splits(self, dataset: DmpDataset) -> tuple[list, list, list, float]:

        graph_list, geometric_scale = self.get_dataset(dataset)

        train_size, test_size, val_size = 0.7, 0.15, 0.15

        train_mask, val_mask, test_mask = self._get_splits(graph_list, config.seed, config.stratified_split, train_size, test_size, val_size)

        train_graph_list, val_graph_list, test_graph_list =  self._extract_splits(graph_list, train_mask, val_mask, test_mask)

        return train_graph_list, val_graph_list, test_graph_list, geometric_scale


    def get_dataset(self, dataset: DmpDataset, geometric_scale: float = None) -> tuple[list[Graph], float]:

        graph_list = self.datasetParser.load_parsed_dataset(dataset)

        n_samples = config.fast_run_n_samples if config.fast_run else config.n_samples

        graph_list = self._subsample(graph_list, n_samples)

        geometric_scale = geometric_scale or self._get_dataset_max_coord(graph_list)

        normalized_graph_list = self._normalize_dataset(graph_list, geometric_scale)

        if config.molecule_orientation != 'none':
            return self._orient_dataset(normalized_graph_list, config.molecule_orientation), geometric_scale

        return normalized_graph_list, geometric_scale


    def _subsample(self, graph_list: list[Graph], subsample_size: int) -> list[Graph]:

        if subsample_size > len(graph_list): return graph_list

        random.seed(config.seed)

        subsample = random.sample(graph_list, subsample_size)

        return subsample


    def _extract_splits(self, graph_list, train_mask: list, val_mask: list, test_mask: list) -> tuple[list, list, list]:

        train_graph_list = [graph_list[i] for i in train_mask]
        val_graph_list = [graph_list[i] for i in val_mask]
        test_graph_list = [graph_list[i] for i in test_mask]

        return train_graph_list, val_graph_list, test_graph_list


    def _get_dataset_max_coord(self, graph_list: list[Graph]) -> float:
        stacked = torch.cat([graph.x for graph in graph_list], dim = 0)
        return stacked.max()


    def _normalize_dataset(self, graph_list: list[Graph], geometric_scale) -> list[Graph]:

        for graph in graph_list:
            graph.x = graph.x / geometric_scale

        return graph_list

    def _get_splits(self, graph_list: list[Graph], seed: int, stratified_split: bool, train_size: float, test_size: float, val_size: float) -> tuple[list, list, list]:
        datasetSplitter = DatasetSplitter()

        target_list: list[float] = [graph.y for graph in graph_list]

        splits: list[list]
        if stratified_split:
            splits = datasetSplitter(np.array(target_list), seed, "stratified", train_size, test_size, val_size)
        else:
            splits = datasetSplitter(np.array(target_list), seed, "standard", train_size, test_size, val_size)

        train_mask = splits[0]
        val_mask = splits[1]
        test_mask = splits[2]

        return train_mask, val_mask, test_mask


    def _orient_dataset(self, graph_list: list[Graph], orientation_axis: Literal[ 'x', 'y', 'z']) -> list[Graph]:

        axis_number: int = {'x': 0, 'y': 1, 'z': 2}[orientation_axis]

        for graph in graph_list:
            graph.x = self.orient_graph_to_axis(graph.x, axis_number)

        return graph_list


    def orient_graph_to_axis(self, graph_coordinate_list: Tensor, target_axis: int) -> list[list[float]]:
        source_axis = self._max_spread_axis(graph_coordinate_list)
        if source_axis == target_axis:
            return graph_coordinate_list

        axis_order = self._build_rotation_order(source_axis, target_axis)
        return self._apply_permutation(graph_coordinate_list, axis_order)


    def _max_spread_axis(self, graph_coordinate_list: Tensor) -> int:
        spans = torch.amax(graph_coordinate_list, dim = 0) - torch.amin(graph_coordinate_list, dim = 0)
        return torch.argmax(spans).item()


    def _build_rotation_order(self, source_axis: int, target_axis: int) -> list[int]:
        k = (target_axis - source_axis) % 3
        return [(i - k) % 3 for i in range(3)]


    def _apply_permutation(self, atom_features: Tensor, axis_order: list[int]) -> Tensor:
        return atom_features[:, axis_order]
