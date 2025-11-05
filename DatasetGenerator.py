import pickle

import warnings

from pandas import DataFrame

import numpy as np
from numpy import ndarray

import rdkit
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem import Mol
from rdkit.Chem import AllChem


import torch
from torch.linalg import norm

from torch_geometric.data import Data as Graph

from DatasetSplitter import DatasetSplitter
from DmiConfig import DmiConfig
from DmpDataset import DmpDataset

from pathlib import Path


class DatasetGenerator:

    def get_dataset(self, config: DmiConfig, add_hydrogens: bool = True) -> tuple[list, list, list, float]:

        graph_list, geometric_scale = self.get_whole_dataset(config, add_hydrogens)

        train_size, test_size, val_size = 0.7, 0.15, 0.15

        train_mask, val_mask, test_mask = self._get_splits(graph_list, config.seed, config.stratified_split, train_size, test_size, val_size)

        train_graph_list, val_graph_list, test_graph_list =  self._extract_splits(graph_list, train_mask, val_mask, test_mask)

        return train_graph_list, val_graph_list, test_graph_list, geometric_scale


    def get_whole_dataset(self, config: DmiConfig, add_hydrogens: bool, geometric_scale: float = None) -> tuple[list[Graph], float]:

        data = self._load_from_file(config)

        graph_list: list[Graph] = self._parse_to_graph_list(data, config, add_hydrogens)

        geometric_scale = geometric_scale or self._get_dataset_max_coord(graph_list)

        normalized_graph_list = self._normalize_dataset(graph_list, geometric_scale)

        return normalized_graph_list, geometric_scale


    def _extract_splits(self, graph_list, train_mask: list, val_mask: list, test_mask: list) -> tuple[list, list, list]:

        train_graph_list = [graph_list[i] for i in train_mask]
        val_graph_list = [graph_list[i] for i in val_mask]
        test_graph_list = [graph_list[i] for i in test_mask]

        return train_graph_list, val_graph_list, test_graph_list


    def _parse_to_graph_list(self, data: DataFrame, config: DmiConfig, add_hydrogens: bool) -> list[Graph]:

        target_list: ndarray = data["target"]

        molecule_list = self._extract_molecules_descriptors(data)

        graph_list: list[Graph] = []

        RDLogger.DisableLog('rdApp.*')

        for molecule, target in zip(molecule_list, target_list):
            try:
                molecule = self._mol_to_graph(molecule, config, target, add_hydrogens)
                graph_list.append(molecule)
            except ValueError:
                warnings.warn(f"Could not parse to graph molecule with SMILE {Chem.MolToSmiles(molecule)}")

        return graph_list


    def _load_from_file(self, config: DmiConfig) -> DataFrame:

        n_samples = config.fast_run_n_samples if config.fast_run else config.n_samples

        with open(self._get_dataset_path(config.dataset), 'rb') as file:
            data: DataFrame = pickle.load(file)

        samples: int = min(len(data.index), n_samples)

        if samples < n_samples:
            warnings.warn(f"Number of samples requested {n_samples} was higher than the available amount {samples}")

        data = data.sample(n=samples, random_state=2025).reset_index()
        data = data[~data["smiles"].isna()].reset_index()

        return data


    def _get_dataset_path(self, dataset: DmpDataset) -> str:
        base = base = Path(__file__).resolve().parent
        return f"{base}/data/features/{dataset.value}/{dataset.value}_2DAP.pkl"


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


    def _extract_molecules_descriptors(self, data: DataFrame) -> list[Mol]:
        molecule_list = []

        for smile in data["smiles"]:
            molecule = rdkit.Chem.MolFromSmiles(smile)
            molecule_list.append(molecule)

        return molecule_list


    def _mol_to_graph(self, molecule: Mol, config: DmiConfig, y: float, include_hydrogens: bool) -> Graph:

        if(include_hydrogens):
            molecule = rdkit.Chem.AddHs(molecule)

        AllChem.EmbedMolecule(molecule)

        atom_coordinates = []
        edge_index = []

        for atom in molecule.GetAtoms():
            positions = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
            atom_coordinates.append([positions.x, positions.y, positions.z])

        for bond in molecule.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.append((i, j))
            edge_index.append((j, i))

        if config.orient_molecules:
            atom_coordinates = self.orient_to_axis(atom_coordinates, 2)

        return Graph(
                x=torch.tensor(atom_coordinates, dtype=torch.float, device = config.device),
                edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                y = y
            )

    def orient_to_axis(self, atom_features: list[list[float]], target_axis: int) -> list[list[float]]:
        source_axis = self._max_spread_axis(atom_features)
        if source_axis == target_axis:
            return atom_features

        axis_order = self._build_rotation_order(source_axis, target_axis)
        return self._apply_permutation(atom_features, axis_order)


    def _max_spread_axis(self, atom_features: list[list[float]]) -> int:
        spans = [(max(a[i] for a in atom_features) - min(a[i] for a in atom_features))
                for i in range(3)]
        return spans.index(max(spans))


    def _build_rotation_order(self, source_axis: int, target_axis: int) -> list[int]:
        k = (target_axis - source_axis) % 3
        return [(i - k) % 3 for i in range(3)]


    def _apply_permutation(self, atom_features: list[list[float]], axis_order: list[int]) -> list[list[float]]:
        return [
            [coords[axis_order[0]], coords[axis_order[1]], coords[axis_order[2]]]
            for coords in atom_features
        ]

